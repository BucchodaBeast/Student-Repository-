"""
Launchpad — Student Project Repository
---------------------------------------
Flask backend with:
  - Role-based accounts (Student, Researcher, Admin) with session auth
  - Admin-gated project approval workflow (pending / approved / rejected)
  - SQLite persistence
  - TF-IDF based semantic search (finds conceptually related projects, not just keyword matches)
  - Cosine-similarity plagiarism / overlap detection on submission
  - "Related projects" recommendations per project page
  - Live-system links + repo links + status tracking
  - Per-project Q&A / discussion threads
  - Admin analytics dashboard (Ledger)
"""

import os
import re
import requests
import psycopg2
import psycopg2.extras
from datetime import datetime
from functools import wraps
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, g, abort, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.local import LocalProxy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ai

try:
    from authlib.integrations.flask_client import OAuth
    AUTHLIB_AVAILABLE = True
except ImportError:
    AUTHLIB_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
SIMILARITY_FLAG_THRESHOLD = 0.15  # cosine similarity above this triggers an overlap warning
# Note: with TF-IDF over short abstracts, meaningful overlap tends to land in the
# 0.15-0.30 range rather than the 0.5+ you'd see with longer documents or embeddings.

# Postgres/Supabase connection string. Use Supabase's "Transaction pooler"
# connection string (port 6543), not the direct connection (port 5432) —
# each web request opens its own connection, and the pooler is what keeps
# that from exhausting Supabase's direct-connection limit under load.
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")


class PGCursorResult:
    """Wraps a psycopg2 cursor so call sites written against sqlite3's
    Connection.execute() shortcut — row["col"] access via fetchone/fetchall,
    plus .lastrowid after an insert — keep working unchanged against Postgres."""

    def __init__(self, cursor, lastrowid=None):
        self._cursor = cursor
        self.lastrowid = lastrowid

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


class PGConnection:
    """Thin wrapper around a psycopg2 connection providing sqlite3-style
    `db.execute(sql, params)` (with `?` placeholders) so the rest of the
    app didn't need a line-by-line query rewrite to move off SQLite."""

    def __init__(self, dsn):
        self._conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)

    def execute(self, sql, params=()):
        pg_sql = re.sub(r"\?", "%s", sql)
        cur = self._conn.cursor()
        cur.execute(pg_sql, params)
        lastrowid = None
        if "returning" in pg_sql.lower():
            row = cur.fetchone()
            lastrowid = row["id"] if row else None
        return PGCursorResult(cur, lastrowid)

    def executescript(self, script):
        cur = self._conn.cursor()
        for statement in filter(None, (s.strip() for s in script.split(";"))):
            cur.execute(statement)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


def get_standalone_connection():
    """A PGConnection outside of a Flask request/app context — for one-off
    scripts like seed.py that need the database without `g`."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL (or SUPABASE_DB_URL) is not set. Launchpad runs on "
            "Postgres/Supabase now — set this to your Supabase connection string "
            "(Project Settings -> Database -> Connection string -> Transaction pooler)."
        )
    return PGConnection(DATABASE_URL)

ROLES = ["student", "researcher", "admin"]
PUBLIC_ROLES = ["student", "researcher"]  # roles selectable without an invite code
STATUSES = ["Ongoing", "Completed", "Archived"]
APPROVALS_NEEDED = 2   # distinct admins must approve before a pending project goes live
REJECTIONS_NEEDED = 2  # distinct admins must reject before a pending project is turned away
ADMIN_INVITE_CODE = os.environ.get("ADMIN_INVITE_CODE", "LAUNCHPAD-ADMIN-2026")

# Curated so "Most Used Technologies" tallies real tech terms out of the free-text
# keywords field rather than every noun someone typed in.
KNOWN_TECHNOLOGIES = [
    "java", "python", "javascript", "typescript", "flutter", "dart", "react", "vue", "angular",
    "spring boot", "spring", "django", "flask", "node.js", "node", "express", "firebase",
    "mysql", "postgresql", "sqlite", "mongodb", "tensorflow", "pytorch", "scikit-learn",
    "keras", "opencv", "pandas", "numpy", "docker", "kubernetes", "aws", "azure", "gcp",
    "html", "css", "tailwind", "bootstrap", "swift", "kotlin", "c++", "c#", "php", "laravel",
    "graphql", "rest api", "blockchain", "solidity", "arduino", "raspberry pi", "r",
    "machine learning", "deep learning", "nlp", "computer vision",
]

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_OAUTH_ENABLED = bool(AUTHLIB_AVAILABLE and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# ---------------------------------------------------------------------------
# Report file storage — Supabase Storage in production, local disk otherwise.
#
# Render's filesystem is wiped on every redeploy, so files saved to UPLOAD_DIR
# don't survive a deploy any more than the old SQLite file did. If SUPABASE_URL
# and SUPABASE_SERVICE_KEY are set, uploads go to a Supabase Storage bucket
# instead and downloads are served via short-lived signed URLs (the bucket
# should be PRIVATE — signed URLs are what let pending/rejected projects stay
# gated behind the existing approval check instead of being world-readable).
# Falls back to local disk when unset, so local dev doesn't need a bucket.
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "reports")
SUPABASE_STORAGE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def upload_report_to_supabase(file_storage, filename):
    """Uploads a report file to Supabase Storage. Returns the stored filename
    on success, or None if Supabase Storage isn't configured or the call fails
    (caller should fall back gracefully rather than crash the submission)."""
    if not SUPABASE_STORAGE_ENABLED:
        return None
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{filename}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": file_storage.mimetype or "application/octet-stream",
        "x-upsert": "true",
    }
    try:
        resp = requests.post(url, headers=headers, data=file_storage.read(), timeout=20)
        resp.raise_for_status()
        return filename
    except Exception:
        return None


def get_signed_report_url(filename, expires_in=60):
    """Returns a short-lived signed download URL for a report already in
    Supabase Storage, or None on failure."""
    if not SUPABASE_STORAGE_ENABLED or not filename:
        return None
    url = f"{SUPABASE_URL}/storage/v1/object/sign/{SUPABASE_STORAGE_BUCKET}/{filename}"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    try:
        resp = requests.post(url, headers=headers, json={"expiresIn": expires_in}, timeout=10)
        resp.raise_for_status()
        signed_path = resp.json().get("signedURL")
        return f"{SUPABASE_URL}/storage/v1{signed_path}" if signed_path else None
    except Exception:
        return None

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

oauth = None
if GOOGLE_OAUTH_ENABLED:
    oauth = OAuth(app)
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

@app.context_processor
def inject_globals():
    return {
        "ai_enabled": ai.ai_enabled(),
        "current_year": datetime.now().year,
        "current_user": current_user,
        "google_oauth_enabled": GOOGLE_OAUTH_ENABLED,
    }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = get_standalone_connection()
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


DEPARTMENTS = [
    "Computer Science",
    "Nursing Sciences",
    "Radiography",
    "Medical Laboratory",
    "Social Work",
    "Psychology",
    "Pharmacy",
]

# Minimal inline SVG line-icons (24x24, stroke=currentColor) — one per department.
DEPARTMENT_ICONS = {
    "Computer Science": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="3" y="4" width="18" height="12" rx="1.5"/><path d="M8 20h8M12 16v4" stroke-linecap="round"/></svg>',
    "Nursing Sciences": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M12 3v8M8 7h8" stroke-linecap="round"/><circle cx="12" cy="16" r="5"/></svg>',
    "Radiography": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="12" cy="12" r="9"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M6.3 6.3l2 2M15.7 15.7l2 2M6.3 17.7l2-2M15.7 8.3l2-2" stroke-linecap="round"/><circle cx="12" cy="12" r="2.5"/></svg>',
    "Medical Laboratory": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M9 2v7.5L4.5 18a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L15 9.5V2" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 2h10M6.5 15h11" stroke-linecap="round"/></svg>',
    "Social Work": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="8" cy="8" r="3"/><circle cx="17" cy="9" r="2.5"/><path d="M3 20v-1a5 5 0 0 1 5-5h1a5 5 0 0 1 4.9 4M15 20v-.8a4 4 0 0 1 4-4h.5" stroke-linecap="round"/></svg>',
    "Psychology": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M9 2a5 5 0 0 0-3 9 6 6 0 0 0 3 11h3a1 1 0 0 0 1-1V4a2 2 0 0 0-2-2H9Z" stroke-linejoin="round"/><path d="M14 6a3 3 0 0 1 3 3v2a4 4 0 0 1 0 8" stroke-linecap="round"/></svg>',
    "Pharmacy": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M6 3h6l5 5v2M6 3v6l-2.5 8a2 2 0 0 0 1.9 2.7h9.2a2 2 0 0 0 1.9-2.7L14 11" stroke-linecap="round" stroke-linejoin="round"/><path d="M6 11h8" stroke-linecap="round"/></svg>',
}


def init_db():
    db = get_standalone_connection()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            student_name TEXT NOT NULL,
            supervisor TEXT NOT NULL,
            keywords TEXT NOT NULL,
            topic_area TEXT NOT NULL,
            department TEXT NOT NULL DEFAULT 'Computer Science',
            institution TEXT NOT NULL DEFAULT 'Eswatini Medical Christian University',
            year INTEGER NOT NULL,
            file_path TEXT,
            summary TEXT,
            builds_on_id INTEGER,
            owner_id INTEGER,
            live_demo_url TEXT,
            github_url TEXT,
            status TEXT NOT NULL DEFAULT 'Ongoing',
            approval_status TEXT NOT NULL DEFAULT 'pending',
            review_note TEXT,
            reviewed_by INTEGER,
            reviewed_at TEXT,
            view_count INTEGER NOT NULL DEFAULT 0,
            download_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (builds_on_id) REFERENCES projects (id),
            FOREIGN KEY (owner_id) REFERENCES users (id),
            FOREIGN KEY (reviewed_by) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS comments (
            id SERIAL PRIMARY KEY,
            project_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            parent_id INTEGER,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (parent_id) REFERENCES comments (id)
        );

        CREATE TABLE IF NOT EXISTS overlap_flags (
            id SERIAL PRIMARY KEY,
            project_id INTEGER NOT NULL,
            similar_project_id INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (similar_project_id) REFERENCES projects (id)
        );

        CREATE TABLE IF NOT EXISTS search_log (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            searched_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_reviews (
            id SERIAL PRIMARY KEY,
            project_id INTEGER NOT NULL,
            admin_id INTEGER NOT NULL,
            decision TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(project_id, admin_id),
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (admin_id) REFERENCES users (id)
        );
        """
    )
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Auth: user model + login manager
# ---------------------------------------------------------------------------

class AuthUser:
    def __init__(self, row):
        self.id = str(row["id"])
        self.name = row["name"]
        self.email = row["email"]
        self.role = row["role"]
        self.is_authenticated = True

    @property
    def role_label(self):
        return {"student": "Student", "researcher": "Researcher", "admin": "Admin"}.get(self.role, self.role)

    @property
    def is_admin(self):
        return self.role == "admin"


class AnonymousUser:
    is_authenticated = False
    is_admin = False
    id = None
    name = ""
    role = None
    role_label = ""


def _load_logged_in_user():
    if "user" in g:
        return g.user
    user_id = session.get("user_id")
    g.user = AnonymousUser()
    if user_id:
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            g.user = AuthUser(row)
    return g.user


current_user = LocalProxy(_load_logged_in_user)


def login_user(row):
    session["user_id"] = row["id"]
    session.permanent = True


def logout_user():
    session.pop("user_id", None)


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


def admin_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Only admins can access that page.", "error")
            return redirect(url_for("index"))
        return view_func(*args, **kwargs)
    return wrapped


def can_manage_project(project_row):
    if not current_user.is_authenticated:
        return False
    if current_user.is_admin:
        return True
    return project_row["owner_id"] is not None and str(project_row["owner_id"]) == current_user.id


app.jinja_env.globals["can_manage_project"] = can_manage_project


# ---------------------------------------------------------------------------
# Routes: auth
# ---------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "student")
        invite_code = request.form.get("invite_code", "").strip()

        if not all([name, email, password]) or role not in ROLES:
            flash("Please fill in every field.", "error")
            return render_template("register.html", form=request.form, roles=ROLES)

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html", form=request.form, roles=ROLES)

        # Admin can be picked directly, but only ever creates an account when
        # paired with a valid invite code — no invite code, no admin account.
        if role == "admin":
            if not invite_code or invite_code != ADMIN_INVITE_CODE:
                flash("A valid admin invite code is required to create an admin account.", "error")
                return render_template("register.html", form=request.form, roles=ROLES)

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("An account with that email already exists.", "error")
            return render_template("register.html", form=request.form, roles=ROLES)

        cur = db.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?) RETURNING id",
            (name, email, generate_password_hash(password), role, datetime.now().isoformat()),
        )
        db.commit()
        row = db.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        login_user(row)
        flash(f"Welcome to Launchpad, {name.split()[0]}.", "success")
        return redirect(url_for("index"))

    return render_template("register.html", form={}, roles=ROLES)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            login_user(row)
            flash(f"Welcome back, {row['name'].split()[0]}.", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("index"))
        flash("Incorrect email or password.", "error")
        return render_template("login.html", form=request.form)

    return render_template("login.html", form={})


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Signed out.", "success")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Google sign-in — Student/Researcher only. A Google login can never grant the
# admin role: brand-new sign-ins always land as 'student' unless the person
# explicitly picks 'researcher' on the one-time completion step below, and an
# existing account keeps whatever role it already had (which, for anyone
# already an admin, was only ever granted through the invite-code path).
# ---------------------------------------------------------------------------

@app.route("/auth/google")
def google_login():
    if not GOOGLE_OAUTH_ENABLED:
        flash("Google sign-in isn't configured on this server yet.", "error")
        return redirect(url_for("login"))
    redirect_uri = url_for("google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    if not GOOGLE_OAUTH_ENABLED:
        flash("Google sign-in isn't configured on this server yet.", "error")
        return redirect(url_for("login"))

    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo")
        if not userinfo:
            userinfo = oauth.google.parse_id_token(token, nonce=session.get("google_nonce"))
    except Exception:
        flash("Google sign-in failed — please try again.", "error")
        return redirect(url_for("login"))

    email = (userinfo.get("email") or "").strip().lower()
    name = userinfo.get("name") or (email.split("@")[0] if email else "")
    if not email:
        flash("Google didn't share an email address — can't sign you in.", "error")
        return redirect(url_for("login"))

    db = get_db()
    row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row:
        login_user(row)
        flash(f"Welcome back, {row['name'].split()[0]}.", "success")
        return redirect(url_for("index"))

    # Brand-new account — one extra step to choose Student or Researcher.
    # This path can never produce an admin account.
    session["pending_google_signup"] = {"email": email, "name": name}
    return redirect(url_for("complete_google_signup"))


@app.route("/complete-signup", methods=["GET", "POST"])
def complete_google_signup():
    pending = session.get("pending_google_signup")
    if not pending:
        flash("That signup link has expired — please try again.", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        role = request.form.get("role", "student")
        if role not in PUBLIC_ROLES:
            flash("Please choose Student or Researcher.", "error")
            return render_template("complete_google_signup.html", pending=pending, roles=PUBLIC_ROLES)

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = ?", (pending["email"],)).fetchone()
        if existing:
            login_user(existing)
            session.pop("pending_google_signup", None)
            return redirect(url_for("index"))

        # Google-authenticated accounts don't need a usable local password.
        unusable_password = generate_password_hash(os.urandom(24).hex())
        cur = db.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?) RETURNING id",
            (pending["name"], pending["email"], unusable_password, role, datetime.now().isoformat()),
        )
        db.commit()
        row = db.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        login_user(row)
        session.pop("pending_google_signup", None)
        flash(f"Welcome to Launchpad, {pending['name'].split()[0]}.", "success")
        return redirect(url_for("index"))

    return render_template("complete_google_signup.html", pending=pending, roles=PUBLIC_ROLES)


# ---------------------------------------------------------------------------
# Text intelligence: TF-IDF corpus, similarity, search, auto-summary
# ---------------------------------------------------------------------------

def get_all_projects_raw(approved_only=False):
    db = get_db()
    if approved_only:
        return db.execute(
            "SELECT * FROM projects WHERE approval_status = 'approved' ORDER BY created_at DESC"
        ).fetchall()
    return db.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()


def build_corpus(approved_only=False):
    """Returns (project_rows, tfidf_matrix, vectorizer) over title+abstract+keywords."""
    rows = get_all_projects_raw(approved_only=approved_only)
    if not rows:
        return [], None, None
    documents = [f"{r['title']} {r['abstract']} {r['keywords']}" for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return rows, None, None
    return rows, matrix, vectorizer


def semantic_search(query, top_n=10, approved_only=True):
    rows, matrix, vectorizer = build_corpus(approved_only=approved_only)
    if matrix is None or not query.strip():
        return []
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)
    return [(r, s) for r, s in ranked if s > 0.05][:top_n]


def find_similar_projects(title, abstract, keywords, exclude_id=None, top_n=5, approved_only=False):
    """Used both for plagiarism-style overlap flagging (approved_only=False, sees
    everything on file) and 'related projects' recommendations (approved_only=True,
    so nobody is pointed at a project that hasn't been approved yet)."""
    rows, matrix, vectorizer = build_corpus(approved_only=approved_only)
    if matrix is None:
        return []
    query_doc = f"{title} {abstract} {keywords}"
    query_vec = vectorizer.transform([query_doc])
    scores = cosine_similarity(query_vec, matrix).flatten()
    results = []
    for row, score in zip(rows, scores):
        if exclude_id is not None and row["id"] == exclude_id:
            continue
        results.append((row, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def auto_summary(abstract, max_sentences=2):
    """Lightweight extractive summary (first N informative sentences).
    Swap this for a Groq/Claude API call in production for true abstractive summaries."""
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    sentences = [s for s in sentences if len(s) > 20]
    return " ".join(sentences[:max_sentences]) if sentences else abstract[:180]


# ---------------------------------------------------------------------------
# Routes: core browsing
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    query = request.args.get("q", "").strip()
    dept_filter = request.args.get("department", "").strip()
    db = get_db()

    if query:
        db.execute(
            "INSERT INTO search_log (query, searched_at) VALUES (?, ?)",
            (query.lower(), datetime.now().isoformat()),
        )
        db.commit()
        results = semantic_search(query, top_n=20, approved_only=True)
        projects = [r for r, s in results]
        if dept_filter:
            projects = [p for p in projects if p["department"] == dept_filter]
        scores = {r["id"]: round(s * 100) for r, s in results}
    else:
        base_sql = "SELECT * FROM projects WHERE approval_status = 'approved'"
        params = []
        if dept_filter:
            base_sql += " AND department = ?"
            params.append(dept_filter)
        base_sql += " ORDER BY created_at DESC"
        projects = db.execute(base_sql, params).fetchall()
        scores = {}

    dept_counts_rows = db.execute(
        "SELECT department, COUNT(*) as c FROM projects WHERE approval_status = 'approved' GROUP BY department ORDER BY c DESC"
    ).fetchall()
    dept_counts = {r["department"]: r["c"] for r in dept_counts_rows}

    comment_counts_rows = db.execute(
        "SELECT project_id, COUNT(*) as c FROM comments GROUP BY project_id"
    ).fetchall()
    comment_counts = {r["project_id"]: r["c"] for r in comment_counts_rows}

    return render_template(
        "index.html",
        projects=projects,
        query=query,
        scores=scores,
        dept_counts=dept_counts,
        dept_icons=DEPARTMENT_ICONS,
        active_department=dept_filter,
        comment_counts=comment_counts,
    )


@app.route("/profile")
@login_required
def profile():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM projects WHERE owner_id = ? ORDER BY created_at DESC", (current_user.id,)
    ).fetchall()
    user_row = db.execute("SELECT * FROM users WHERE id = ?", (current_user.id,)).fetchone()
    return render_template("profile.html", projects=rows, member=user_row)


@app.route("/project/<int:project_id>")
@login_required
def project_detail(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("index"))

    if project["approval_status"] != "approved" and not can_manage_project(project):
        flash("Project not found.", "error")
        return redirect(url_for("index"))

    db.execute("UPDATE projects SET view_count = view_count + 1 WHERE id = ?", (project_id,))
    db.commit()

    related = find_similar_projects(
        project["title"], project["abstract"], project["keywords"], exclude_id=project_id, top_n=4, approved_only=True
    )
    related = [(r, round(s * 100)) for r, s in related if s > 0.05]

    builds_on = None
    if project["builds_on_id"]:
        builds_on = db.execute(
            "SELECT * FROM projects WHERE id = ?", (project["builds_on_id"],)
        ).fetchone()

    lineage_children = db.execute(
        "SELECT * FROM projects WHERE builds_on_id = ?", (project_id,)
    ).fetchall()

    owner = None
    if project["owner_id"]:
        owner = db.execute("SELECT * FROM users WHERE id = ?", (project["owner_id"],)).fetchone()

    reviewer = None
    if project["reviewed_by"]:
        reviewer = db.execute("SELECT * FROM users WHERE id = ?", (project["reviewed_by"],)).fetchone()

    questions, answers_by_parent = get_discussion(project_id)

    review_progress = None
    if project["approval_status"] == "pending":
        review_progress = get_review_progress(project_id, current_user.id if current_user.is_authenticated else None)

    return render_template(
        "project_detail.html",
        project=project,
        related=related,
        builds_on=builds_on,
        lineage_children=lineage_children,
        owner=owner,
        reviewer=reviewer,
        questions=questions,
        answers_by_parent=answers_by_parent,
        review_progress=review_progress,
        approvals_needed=APPROVALS_NEEDED,
        rejections_needed=REJECTIONS_NEEDED,
    )


def get_discussion(project_id):
    """Returns (top-level questions, {question_id: [answers]}) for a project,
    each comment row annotated with the poster's name/role."""
    db = get_db()
    rows = db.execute(
        """SELECT comments.*, users.name as author_name, users.role as author_role
           FROM comments JOIN users ON comments.user_id = users.id
           WHERE comments.project_id = ? ORDER BY comments.created_at ASC""",
        (project_id,),
    ).fetchall()
    questions = [r for r in rows if r["parent_id"] is None]
    answers_by_parent = {}
    for r in rows:
        if r["parent_id"] is not None:
            answers_by_parent.setdefault(r["parent_id"], []).append(r)
    return questions, answers_by_parent


@app.route("/project/<int:project_id>/download")
@login_required
def download_report(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project or not project["file_path"]:
        flash("No report file is on file for this project.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    if project["approval_status"] != "approved" and not can_manage_project(project):
        flash("Project not found.", "error")
        return redirect(url_for("index"))

    db.execute("UPDATE projects SET download_count = download_count + 1 WHERE id = ?", (project_id,))
    db.commit()

    if SUPABASE_STORAGE_ENABLED:
        signed_url = get_signed_report_url(project["file_path"])
        if signed_url:
            return redirect(signed_url)
        flash("Couldn't retrieve the report file right now — please try again shortly.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    return send_from_directory(UPLOAD_DIR, project["file_path"], as_attachment=True)


@app.route("/project/<int:project_id>/comment", methods=["POST"])
@login_required
def add_comment(project_id):
    db = get_db()
    project = db.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("index"))

    body = request.form.get("body", "").strip()
    parent_id = request.form.get("parent_id") or None

    if not body:
        flash("Comment can't be empty.", "error")
        return redirect(url_for("project_detail", project_id=project_id) + "#discussion")

    db.execute(
        "INSERT INTO comments (project_id, user_id, parent_id, body, created_at) VALUES (?, ?, ?, ?, ?)",
        (project_id, current_user.id, parent_id, body, datetime.now().isoformat()),
    )
    db.commit()
    flash("Posted." if parent_id is None else "Answer posted.", "success")
    return redirect(url_for("project_detail", project_id=project_id) + "#discussion")


# ---------------------------------------------------------------------------
# Routes: upload / edit with live overlap detection
# ---------------------------------------------------------------------------

def _project_form_fields(form):
    return dict(
        title=form.get("title", "").strip(),
        abstract=form.get("abstract", "").strip(),
        student_name=form.get("student_name", "").strip(),
        supervisor=form.get("supervisor", "").strip(),
        keywords=form.get("keywords", "").strip(),
        topic_area=form.get("topic_area", "").strip(),
        department=form.get("department", "").strip(),
        institution=form.get("institution", "").strip() or "Eswatini Medical Christian University",
        year=form.get("year", "").strip(),
        builds_on_id=form.get("builds_on_id") or None,
        live_demo_url=form.get("live_demo_url", "").strip(),
        github_url=form.get("github_url", "").strip(),
        status=form.get("status", "Ongoing").strip() or "Ongoing",
    )


@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit():
    db = get_db()
    all_projects = db.execute("SELECT id, title FROM projects ORDER BY title").fetchall()

    if request.method == "POST":
        f = _project_form_fields(request.form)
        confirm_override = request.form.get("confirm_override") == "yes"

        required = [f["title"], f["abstract"], f["student_name"], f["supervisor"],
                    f["keywords"], f["topic_area"], f["department"], f["year"]]
        if not all(required):
            flash("Please fill in all required fields.", "error")
            return render_template("submit.html", all_projects=all_projects, form=request.form,
                                   departments=DEPARTMENTS, statuses=STATUSES, mode="create")

        similar = find_similar_projects(f["title"], f["abstract"], f["keywords"], top_n=3)
        flagged = [(r, s) for r, s in similar if s >= SIMILARITY_FLAG_THRESHOLD]

        if flagged and not confirm_override:
            ai_reasoning = ai.ai_overlap_analysis(f["title"], f["abstract"], flagged)
            return render_template(
                "submit.html",
                all_projects=all_projects,
                form=request.form,
                departments=DEPARTMENTS,
                statuses=STATUSES,
                mode="create",
                flagged=[(r, round(s * 100)) for r, s in flagged],
                ai_reasoning=ai_reasoning,
            )

        file_path = None
        file = request.files.get("report_file")
        if file and file.filename and "." in file.filename:
            ext = file.filename.rsplit(".", 1)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
                if SUPABASE_STORAGE_ENABLED:
                    file_path = upload_report_to_supabase(file, filename)
                    if file_path is None:
                        flash("The project saved, but the report file failed to upload — "
                              "you can add it later by editing the project.", "error")
                else:
                    file.save(os.path.join(UPLOAD_DIR, filename))
                    file_path = filename

        summary = ai.ai_summary(f["title"], f["abstract"]) or auto_summary(f["abstract"])

        # Admin uploads skip the review queue since there's no one else to approve them.
        if current_user.is_admin:
            approval_status, reviewed_by, reviewed_at = "approved", current_user.id, datetime.now().isoformat()
        else:
            approval_status, reviewed_by, reviewed_at = "pending", None, None

        cur = db.execute(
            """INSERT INTO projects
               (title, abstract, student_name, supervisor, keywords, topic_area, department,
                institution, year, file_path, summary, builds_on_id, owner_id, live_demo_url,
                github_url, status, approval_status, reviewed_by, reviewed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id""",
            (f["title"], f["abstract"], f["student_name"], f["supervisor"], f["keywords"],
             f["topic_area"], f["department"], f["institution"], int(f["year"]), file_path,
             summary, f["builds_on_id"], current_user.id, f["live_demo_url"] or None,
             f["github_url"] or None, f["status"], approval_status, reviewed_by, reviewed_at,
             datetime.now().isoformat()),
        )
        new_id = cur.lastrowid

        for row, score in flagged:
            db.execute(
                """INSERT INTO overlap_flags (project_id, similar_project_id, similarity_score)
                   VALUES (?, ?, ?)""",
                (new_id, row["id"], score),
            )

        db.commit()
        if current_user.is_admin:
            flash("Project uploaded and approved.", "success")
        else:
            flash("Project submitted — an admin will review it before it appears publicly.", "success")
        return redirect(url_for("project_detail", project_id=new_id))

    return render_template("submit.html", all_projects=all_projects, form={}, departments=DEPARTMENTS,
                           statuses=STATUSES, mode="create")


@app.route("/project/<int:project_id>/edit", methods=["GET", "POST"])
@login_required
def edit_project(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("index"))
    if not can_manage_project(project):
        flash("You can only manage projects you uploaded.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    all_projects = db.execute(
        "SELECT id, title FROM projects WHERE id != ? ORDER BY title", (project_id,)
    ).fetchall()

    if request.method == "POST":
        f = _project_form_fields(request.form)
        required = [f["title"], f["abstract"], f["student_name"], f["supervisor"],
                    f["keywords"], f["topic_area"], f["department"], f["year"]]
        if not all(required):
            flash("Please fill in all required fields.", "error")
            return render_template("submit.html", all_projects=all_projects, form=request.form,
                                   departments=DEPARTMENTS, statuses=STATUSES, mode="edit", project=project)

        db.execute(
            """UPDATE projects SET title=?, abstract=?, student_name=?, supervisor=?, keywords=?,
               topic_area=?, department=?, institution=?, year=?, builds_on_id=?, live_demo_url=?,
               github_url=?, status=? WHERE id=?""",
            (f["title"], f["abstract"], f["student_name"], f["supervisor"], f["keywords"],
             f["topic_area"], f["department"], f["institution"], int(f["year"]), f["builds_on_id"],
             f["live_demo_url"] or None, f["github_url"] or None, f["status"], project_id),
        )
        db.commit()
        flash("Project updated.", "success")
        return redirect(url_for("project_detail", project_id=project_id))

    return render_template("submit.html", all_projects=all_projects, form=dict(project), departments=DEPARTMENTS,
                           statuses=STATUSES, mode="edit", project=project)


@app.route("/project/<int:project_id>/delete", methods=["POST"])
@login_required
def delete_project(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("index"))
    if not can_manage_project(project):
        flash("You can only manage projects you uploaded.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    db.execute("DELETE FROM comments WHERE project_id = ?", (project_id,))
    db.execute("DELETE FROM project_reviews WHERE project_id = ?", (project_id,))
    db.execute("DELETE FROM overlap_flags WHERE project_id = ? OR similar_project_id = ?", (project_id, project_id))
    db.execute("UPDATE projects SET builds_on_id = NULL WHERE builds_on_id = ?", (project_id,))
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    flash("Project deleted.", "success")
    return redirect(url_for("dashboard") if current_user.is_admin else url_for("profile"))


# ---------------------------------------------------------------------------
# Routes: admin review queue (approve / reject pending uploads)
#
# Every admin acts as a supervisor and casts one vote per project. A project
# only goes live once APPROVALS_NEEDED distinct admins have approved it, and
# is turned away once REJECTIONS_NEEDED distinct admins have rejected it.
# Admins can change their vote at any time while the project is still pending.
# ---------------------------------------------------------------------------

def get_review_progress(project_id, viewer_id=None):
    """Vote tally + individual reviews for a pending project.
    Returns {approvals, rejections, reviews, my_vote} where my_vote is the
    viewer's own current decision ('approve' / 'reject' / None)."""
    db = get_db()
    rows = db.execute(
        """SELECT project_reviews.*, users.name as admin_name FROM project_reviews
           JOIN users ON project_reviews.admin_id = users.id
           WHERE project_id = ? ORDER BY created_at ASC""",
        (project_id,),
    ).fetchall()
    approvals = [r for r in rows if r["decision"] == "approve"]
    rejections = [r for r in rows if r["decision"] == "reject"]
    my_vote = next((r["decision"] for r in rows if viewer_id and str(r["admin_id"]) == str(viewer_id)), None)
    return {"approvals": approvals, "rejections": rejections, "reviews": rows, "my_vote": my_vote}


@app.route("/review-queue")
@login_required
@admin_required
def review_queue():
    db = get_db()
    pending = db.execute(
        """SELECT projects.*, users.name as owner_name FROM projects
           LEFT JOIN users ON projects.owner_id = users.id
           WHERE approval_status = 'pending' ORDER BY created_at ASC"""
    ).fetchall()
    progress = {p["id"]: get_review_progress(p["id"], current_user.id) for p in pending}
    recently_reviewed = db.execute(
        """SELECT projects.*, users.name as owner_name, reviewer.name as reviewer_name
           FROM projects
           LEFT JOIN users ON projects.owner_id = users.id
           LEFT JOIN users as reviewer ON projects.reviewed_by = reviewer.id
           WHERE approval_status IN ('approved', 'rejected') AND reviewed_at IS NOT NULL
           ORDER BY reviewed_at DESC LIMIT 8"""
    ).fetchall()
    return render_template(
        "review_queue.html",
        pending=pending,
        progress=progress,
        recently_reviewed=recently_reviewed,
        approvals_needed=APPROVALS_NEEDED,
        rejections_needed=REJECTIONS_NEEDED,
    )


@app.route("/project/<int:project_id>/review", methods=["POST"])
@login_required
@admin_required
def review_project(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("review_queue"))

    if project["approval_status"] != "pending":
        flash(f"'{project['title']}' has already been {project['approval_status']} — voting is closed.", "error")
        return redirect(url_for("review_queue"))

    action = request.form.get("action")
    note = request.form.get("note", "").strip()

    if action not in ("approve", "reject"):
        flash("Unrecognized review action.", "error")
        return redirect(url_for("review_queue"))

    if action == "reject" and not note:
        flash("Please add a short note explaining why this project was rejected.", "error")
        return redirect(url_for("review_queue"))

    now = datetime.now().isoformat()
    # One vote per admin per project — upsert so an admin can change their mind
    # while the project is still pending.
    db.execute(
        """INSERT INTO project_reviews (project_id, admin_id, decision, note, created_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(project_id, admin_id) DO UPDATE SET
             decision = excluded.decision, note = excluded.note, created_at = excluded.created_at""",
        (project_id, current_user.id, action, note or None, now),
    )
    db.commit()

    progress = get_review_progress(project_id)
    approvals, rejections = len(progress["approvals"]), len(progress["rejections"])

    if approvals >= APPROVALS_NEEDED:
        db.execute(
            "UPDATE projects SET approval_status = 'approved', review_note = NULL, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (current_user.id, now, project_id),
        )
        db.commit()
        flash(f"'{project['title']}' is now approved ({approvals}/{APPROVALS_NEEDED} approvals reached).", "success")
    elif rejections >= REJECTIONS_NEEDED:
        combined_note = "; ".join(r["note"] for r in progress["rejections"] if r["note"])
        db.execute(
            "UPDATE projects SET approval_status = 'rejected', review_note = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (combined_note or None, current_user.id, now, project_id),
        )
        db.commit()
        flash(f"'{project['title']}' was rejected ({rejections}/{REJECTIONS_NEEDED} rejections reached).", "success")
    else:
        flash(
            f"Your {action} vote on '{project['title']}' was recorded "
            f"({approvals}/{APPROVALS_NEEDED} approvals, {rejections}/{REJECTIONS_NEEDED} rejections so far).",
            "success",
        )
    return redirect(url_for("review_queue"))


# ---------------------------------------------------------------------------
# Routes: AI research assistant (RAG over the archive)
# ---------------------------------------------------------------------------

@app.route("/ask", methods=["GET", "POST"])
@login_required
def ask():
    answer = None
    sources = []
    question = ""
    ai_attempted_unavailable = False

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            results = semantic_search(question, top_n=6)
            if not results:
                answer = "Nothing in the archive is conceptually close enough to this question yet."
            else:
                ai_answer = ai.ai_research_assistant(question, results)
                if ai_answer is None:
                    answer = None
                    ai_attempted_unavailable = True
                    flash("Ask the Archive needs GROQ_API_KEY set in the environment.", "error")
                else:
                    answer = ai_answer
                sources = [(r, round(s * 100)) for r, s in results]

    return render_template("ask.html", question=question, answer=answer, sources=sources,
                           ai_attempted_unavailable=ai_attempted_unavailable)


# ---------------------------------------------------------------------------
# Routes: supervisor / topic-gap dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
@admin_required
def dashboard():
    db = get_db()

    gap_department = request.args.get("gap_department", "").strip()
    gap_suggestions = None
    gap_ai_unavailable = False
    if gap_department:
        existing = db.execute(
            "SELECT title FROM projects WHERE department = ?", (gap_department,)
        ).fetchall()
        gap_suggestions = ai.ai_gap_suggestions(gap_department, [r["title"] for r in existing])
        if gap_suggestions is None:
            gap_ai_unavailable = True
            flash("AI Research Gap Advisor needs GROQ_API_KEY set in the environment.", "error")

    supervisor_load = db.execute(
        """SELECT supervisor, COUNT(*) as project_count
           FROM projects GROUP BY supervisor ORDER BY project_count DESC"""
    ).fetchall()

    department_counts = db.execute(
        """SELECT department, COUNT(*) as project_count
           FROM projects GROUP BY department ORDER BY project_count DESC"""
    ).fetchall()

    topic_counts = db.execute(
        """SELECT topic_area, COUNT(*) as project_count
           FROM projects GROUP BY topic_area ORDER BY project_count DESC"""
    ).fetchall()

    year_counts = db.execute(
        """SELECT year, COUNT(*) as project_count
           FROM projects GROUP BY year ORDER BY year DESC"""
    ).fetchall()

    total_projects = db.execute("SELECT COUNT(*) as c FROM projects").fetchone()["c"]
    total_users = db.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    total_comments = db.execute("SELECT COUNT(*) as c FROM comments").fetchone()["c"]
    pending_count = db.execute(
        "SELECT COUNT(*) as c FROM projects WHERE approval_status = 'pending'"
    ).fetchone()["c"]
    live_projects = db.execute(
        "SELECT COUNT(*) as c FROM projects WHERE live_demo_url IS NOT NULL AND live_demo_url != ''"
    ).fetchone()["c"]
    total_downloads = db.execute("SELECT COALESCE(SUM(download_count), 0) as c FROM projects").fetchone()["c"]

    # Technology Insights — tally curated tech terms found in each project's keywords.
    all_keywords = db.execute("SELECT keywords FROM projects").fetchall()
    tech_counter = Counter()
    for row in all_keywords:
        text = row["keywords"].lower()
        for tech in KNOWN_TECHNOLOGIES:
            if tech in text:
                tech_counter[tech] += 1
    top_technologies = tech_counter.most_common(8)
    max_tech_count = top_technologies[0][1] if top_technologies else 1

    # Most Searched Keyword — from the running search log.
    search_rows = db.execute(
        "SELECT query, COUNT(*) as c FROM search_log GROUP BY query ORDER BY c DESC LIMIT 8"
    ).fetchall()
    top_searches = [(r["query"], r["c"]) for r in search_rows]
    max_search_count = top_searches[0][1] if top_searches else 1

    flagged_pairs = db.execute(
        """SELECT overlap_flags.similarity_score, p1.id as pid1, p1.title as t1,
                  p2.id as pid2, p2.title as t2
           FROM overlap_flags
           JOIN projects p1 ON overlap_flags.project_id = p1.id
           JOIN projects p2 ON overlap_flags.similar_project_id = p2.id
           ORDER BY overlap_flags.similarity_score DESC LIMIT 10"""
    ).fetchall()

    return render_template(
        "dashboard.html",
        supervisor_load=supervisor_load,
        department_counts=department_counts,
        topic_counts=topic_counts,
        year_counts=year_counts,
        total_projects=total_projects,
        total_users=total_users,
        total_comments=total_comments,
        pending_count=pending_count,
        live_projects=live_projects,
        total_downloads=total_downloads,
        top_technologies=top_technologies,
        max_tech_count=max_tech_count,
        top_searches=top_searches,
        max_search_count=max_search_count,
        flagged_pairs=flagged_pairs,
        departments=DEPARTMENTS,
        gap_department=gap_department,
        gap_suggestions=gap_suggestions,
        gap_ai_unavailable=gap_ai_unavailable,
        admin_invite_code=ADMIN_INVITE_CODE,
    )


# ---------------------------------------------------------------------------
# JSON API (useful for the presentation demo / future frontend swap)
# ---------------------------------------------------------------------------

@app.route("/api/search")
@login_required
def api_search():
    query = request.args.get("q", "")
    results = semantic_search(query, top_n=10)
    return jsonify([
        {
            "id": r["id"],
            "title": r["title"],
            "supervisor": r["supervisor"],
            "topic_area": r["topic_area"],
            "similarity": round(s * 100, 1),
            "summary": r["summary"],
            "status": r["status"],
            "live_demo_url": r["live_demo_url"],
        }
        for r, s in results
    ])


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
else:
    # Under gunicorn, __main__ never runs — make sure the schema exists.
    init_db()

"""
Launchpad — Student Project Repository
---------------------------------------
Flask backend with:
  - Role-based accounts (Student, Researcher, Supervisor) via Flask-Login
  - SQLite persistence
  - TF-IDF based semantic search (finds conceptually related projects, not just keyword matches)
  - Cosine-similarity plagiarism / overlap detection on submission
  - "Related projects" recommendations per project page
  - Live-system links + repo links + status tracking
  - Per-project Q&A / discussion threads
  - Supervisor load + topic-gap dashboard
"""

import os
import sqlite3
import re
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, g, abort, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.local import LocalProxy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "repository.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
SIMILARITY_FLAG_THRESHOLD = 0.15  # cosine similarity above this triggers an overlap warning
# Note: with TF-IDF over short abstracts, meaningful overlap tends to land in the
# 0.15-0.30 range rather than the 0.5+ you'd see with longer documents or embeddings.

ROLES = ["student", "researcher", "supervisor"]
STATUSES = ["Ongoing", "Completed", "Archived"]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.context_processor
def inject_globals():
    return {"ai_enabled": ai.ai_enabled(), "current_year": datetime.now().year, "current_user": current_user}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


DEPARTMENTS = [
    "Computer Science",
    "Pharmacy",
    "Medical Laboratory Sciences",
    "Nursing Science",
    "Psychology",
    "Public Health",
    "Engineering",
    "Business & Accounting",
    "Other",
]

# Minimal inline SVG line-icons (24x24, stroke=currentColor) — one per department.
DEPARTMENT_ICONS = {
    "Computer Science": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="3" y="4" width="18" height="12" rx="1.5"/><path d="M8 20h8M12 16v4" stroke-linecap="round"/></svg>',
    "Pharmacy": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M6 3h6l5 5v2M6 3v6l-2.5 8a2 2 0 0 0 1.9 2.7h9.2a2 2 0 0 0 1.9-2.7L14 11" stroke-linecap="round" stroke-linejoin="round"/><path d="M6 11h8" stroke-linecap="round"/></svg>',
    "Medical Laboratory Sciences": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M9 2v7.5L4.5 18a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L15 9.5V2" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 2h10M6.5 15h11" stroke-linecap="round"/></svg>',
    "Nursing Science": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M12 3v8M8 7h8" stroke-linecap="round"/><circle cx="12" cy="16" r="5"/></svg>',
    "Psychology": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M9 2a5 5 0 0 0-3 9 6 6 0 0 0 3 11h3a1 1 0 0 0 1-1V4a2 2 0 0 0-2-2H9Z" stroke-linejoin="round"/><path d="M14 6a3 3 0 0 1 3 3v2a4 4 0 0 1 0 8" stroke-linecap="round"/></svg>',
    "Public Health": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="12" cy="12" r="9"/><path d="M12 7v10M7 12h10" stroke-linecap="round"/></svg>',
    "Engineering": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M14.7 6.3a4 4 0 0 1-5.4 5.4L4 17l3 3 5.3-5.3a4 4 0 0 1 5.4-5.4L21 6l-3-3-3.3 3.3Z" stroke-linejoin="round"/></svg>',
    "Business & Accounting": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M4 20V10M10 20V4M16 20v-7M20 20H4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "Other": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M12 3l2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z" stroke-linejoin="round"/></svg>',
}


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            student_name TEXT NOT NULL,
            supervisor TEXT NOT NULL,
            keywords TEXT NOT NULL,
            topic_area TEXT NOT NULL,
            department TEXT NOT NULL DEFAULT 'Other',
            institution TEXT NOT NULL DEFAULT 'Eswatini Medical Christian University',
            year INTEGER NOT NULL,
            file_path TEXT,
            summary TEXT,
            builds_on_id INTEGER,
            owner_id INTEGER,
            live_demo_url TEXT,
            github_url TEXT,
            status TEXT NOT NULL DEFAULT 'Ongoing',
            view_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (builds_on_id) REFERENCES projects (id),
            FOREIGN KEY (owner_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            similar_project_id INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (similar_project_id) REFERENCES projects (id)
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
        return {"student": "Student", "researcher": "Researcher", "supervisor": "Supervisor"}.get(self.role, self.role)

    @property
    def is_supervisor(self):
        return self.role == "supervisor"


class AnonymousUser:
    is_authenticated = False
    is_supervisor = False
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


def supervisor_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_supervisor:
            flash("Only supervisors can access that page.", "error")
            return redirect(url_for("index"))
        return view_func(*args, **kwargs)
    return wrapped


def can_manage_project(project_row):
    if not current_user.is_authenticated:
        return False
    if current_user.is_supervisor:
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

        if not all([name, email, password]) or role not in ROLES:
            flash("Please fill in every field.", "error")
            return render_template("register.html", form=request.form, roles=ROLES)

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html", form=request.form, roles=ROLES)

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("An account with that email already exists.", "error")
            return render_template("register.html", form=request.form, roles=ROLES)

        cur = db.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
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
# Text intelligence: TF-IDF corpus, similarity, search, auto-summary
# ---------------------------------------------------------------------------

def get_all_projects_raw():
    db = get_db()
    return db.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()


def build_corpus():
    """Returns (project_rows, tfidf_matrix, vectorizer) over title+abstract+keywords."""
    rows = get_all_projects_raw()
    if not rows:
        return [], None, None
    documents = [f"{r['title']} {r['abstract']} {r['keywords']}" for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return rows, None, None
    return rows, matrix, vectorizer


def semantic_search(query, top_n=10):
    rows, matrix, vectorizer = build_corpus()
    if matrix is None or not query.strip():
        return []
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)
    return [(r, s) for r, s in ranked if s > 0.05][:top_n]


def find_similar_projects(title, abstract, keywords, exclude_id=None, top_n=5):
    """Used both for plagiarism-style overlap flagging and 'related projects'."""
    rows, matrix, vectorizer = build_corpus()
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
        results = semantic_search(query, top_n=20)
        projects = [r for r, s in results]
        if dept_filter:
            projects = [p for p in projects if p["department"] == dept_filter]
        scores = {r["id"]: round(s * 100) for r, s in results}
    else:
        base_sql = "SELECT * FROM projects"
        params = []
        if dept_filter:
            base_sql += " WHERE department = ?"
            params.append(dept_filter)
        base_sql += " ORDER BY created_at DESC"
        projects = db.execute(base_sql, params).fetchall()
        scores = {}

    dept_counts_rows = db.execute(
        "SELECT department, COUNT(*) as c FROM projects GROUP BY department ORDER BY c DESC"
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


@app.route("/my-projects")
@login_required
def my_projects():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM projects WHERE owner_id = ? ORDER BY created_at DESC", (current_user.id,)
    ).fetchall()
    return render_template("my_projects.html", projects=rows)


@app.route("/project/<int:project_id>")
@login_required
def project_detail(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("index"))

    db.execute("UPDATE projects SET view_count = view_count + 1 WHERE id = ?", (project_id,))
    db.commit()

    related = find_similar_projects(
        project["title"], project["abstract"], project["keywords"], exclude_id=project_id, top_n=4
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

    questions, answers_by_parent = get_discussion(project_id)

    return render_template(
        "project_detail.html",
        project=project,
        related=related,
        builds_on=builds_on,
        lineage_children=lineage_children,
        defense_questions=None,
        owner=owner,
        questions=questions,
        answers_by_parent=answers_by_parent,
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


@app.route("/project/<int:project_id>/defense-prep", methods=["POST"])
@login_required
def defense_prep(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("index"))

    questions_ai = ai.ai_defense_questions(project["title"], project["abstract"], project["department"])
    if questions_ai is None:
        flash("AI defense prep needs GROQ_API_KEY set in the environment.", "error")

    related = find_similar_projects(
        project["title"], project["abstract"], project["keywords"], exclude_id=project_id, top_n=4
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

    questions, answers_by_parent = get_discussion(project_id)

    return render_template(
        "project_detail.html",
        project=project,
        related=related,
        builds_on=builds_on,
        lineage_children=lineage_children,
        defense_questions=questions_ai,
        owner=owner,
        questions=questions,
        answers_by_parent=answers_by_parent,
    )


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
                file.save(os.path.join(UPLOAD_DIR, filename))
                file_path = filename

        summary = ai.ai_summary(f["title"], f["abstract"]) or auto_summary(f["abstract"])

        cur = db.execute(
            """INSERT INTO projects
               (title, abstract, student_name, supervisor, keywords, topic_area, department,
                institution, year, file_path, summary, builds_on_id, owner_id, live_demo_url,
                github_url, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f["title"], f["abstract"], f["student_name"], f["supervisor"], f["keywords"],
             f["topic_area"], f["department"], f["institution"], int(f["year"]), file_path,
             summary, f["builds_on_id"], current_user.id, f["live_demo_url"] or None,
             f["github_url"] or None, f["status"], datetime.now().isoformat()),
        )
        new_id = cur.lastrowid

        for row, score in flagged:
            db.execute(
                """INSERT INTO overlap_flags (project_id, similar_project_id, similarity_score)
                   VALUES (?, ?, ?)""",
                (new_id, row["id"], score),
            )

        db.commit()
        flash("Project uploaded successfully.", "success")
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
    db.execute("DELETE FROM overlap_flags WHERE project_id = ? OR similar_project_id = ?", (project_id, project_id))
    db.execute("UPDATE projects SET builds_on_id = NULL WHERE builds_on_id = ?", (project_id,))
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    flash("Project deleted.", "success")
    return redirect(url_for("my_projects") if not current_user.is_supervisor else url_for("dashboard"))


# ---------------------------------------------------------------------------
# Routes: AI research assistant (RAG over the archive)
# ---------------------------------------------------------------------------

@app.route("/ask", methods=["GET", "POST"])
@login_required
def ask():
    answer = None
    sources = []
    question = ""

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
                    flash("Ask the Archive needs GROQ_API_KEY set in the environment.", "error")
                else:
                    answer = ai_answer
                sources = [(r, round(s * 100)) for r, s in results]

    return render_template("ask.html", question=question, answer=answer, sources=sources)


# ---------------------------------------------------------------------------
# Routes: supervisor / topic-gap dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
@supervisor_required
def dashboard():
    db = get_db()

    gap_department = request.args.get("gap_department", "").strip()
    gap_suggestions = None
    if gap_department:
        existing = db.execute(
            "SELECT title FROM projects WHERE department = ?", (gap_department,)
        ).fetchall()
        gap_suggestions = ai.ai_gap_suggestions(gap_department, [r["title"] for r in existing])
        if gap_suggestions is None:
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
    live_projects = db.execute(
        "SELECT COUNT(*) as c FROM projects WHERE live_demo_url IS NOT NULL AND live_demo_url != ''"
    ).fetchone()["c"]

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
        live_projects=live_projects,
        flagged_pairs=flagged_pairs,
        departments=DEPARTMENTS,
        gap_department=gap_department,
        gap_suggestions=gap_suggestions,
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

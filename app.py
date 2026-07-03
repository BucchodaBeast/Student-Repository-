"""
Student Final-Year Project Repository
--------------------------------------
Flask backend with:
  - SQLite persistence
  - TF-IDF based semantic search (finds conceptually related projects, not just keyword matches)
  - Cosine-similarity plagiarism / overlap detection on submission
  - "Related projects" recommendations per project page
  - Supervisor load + topic-gap dashboard
"""

import os
import sqlite3
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, g
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "repository.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
SIMILARITY_FLAG_THRESHOLD = 0.15  # cosine similarity above this triggers an overlap warning
# Note: with TF-IDF over short abstracts, meaningful overlap tends to land in the
# 0.15-0.30 range rather than the 0.5+ you'd see with longer documents or embeddings.

app = Flask(__name__)
app.secret_key = "dev-secret-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)


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

DEPARTMENT_ICONS = {
    "Computer Science": "&#9711;",
    "Pharmacy": "&#9877;",
    "Medical Laboratory Sciences": "&#9879;",
    "Nursing Science": "&#10010;",
    "Psychology": "&#9685;",
    "Public Health": "&#9906;",
    "Engineering": "&#9881;",
    "Business & Accounting": "&#9878;",
    "Other": "&#10022;",
}


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
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
            created_at TEXT NOT NULL,
            FOREIGN KEY (builds_on_id) REFERENCES projects (id)
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

    return render_template(
        "index.html",
        projects=projects,
        query=query,
        scores=scores,
        dept_counts=dept_counts,
        dept_icons=DEPARTMENT_ICONS,
        active_department=dept_filter,
    )


@app.route("/project/<int:project_id>")
def project_detail(project_id):
    db = get_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("index"))

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

    return render_template(
        "project_detail.html",
        project=project,
        related=related,
        builds_on=builds_on,
        lineage_children=lineage_children,
    )


# ---------------------------------------------------------------------------
# Routes: submission with live overlap detection
# ---------------------------------------------------------------------------

@app.route("/submit", methods=["GET", "POST"])
def submit():
    db = get_db()
    all_projects = db.execute("SELECT id, title FROM projects ORDER BY title").fetchall()

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        abstract = request.form.get("abstract", "").strip()
        student_name = request.form.get("student_name", "").strip()
        supervisor = request.form.get("supervisor", "").strip()
        keywords = request.form.get("keywords", "").strip()
        topic_area = request.form.get("topic_area", "").strip()
        department = request.form.get("department", "").strip()
        institution = request.form.get("institution", "").strip() or "Eswatini Medical Christian University"
        year = request.form.get("year", "").strip()
        builds_on_id = request.form.get("builds_on_id") or None
        confirm_override = request.form.get("confirm_override") == "yes"

        if not all([title, abstract, student_name, supervisor, keywords, topic_area, department, year]):
            flash("Please fill in all required fields.", "error")
            return render_template("submit.html", all_projects=all_projects, form=request.form, departments=DEPARTMENTS)

        # Check overlap BEFORE inserting, unless the student already confirmed
        similar = find_similar_projects(title, abstract, keywords, top_n=3)
        flagged = [(r, s) for r, s in similar if s >= SIMILARITY_FLAG_THRESHOLD]

        if flagged and not confirm_override:
            return render_template(
                "submit.html",
                all_projects=all_projects,
                form=request.form,
                departments=DEPARTMENTS,
                flagged=[(r, round(s * 100)) for r, s in flagged],
            )

        file_path = None
        file = request.files.get("report_file")
        if file and file.filename and "." in file.filename:
            ext = file.filename.rsplit(".", 1)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
                file.save(os.path.join(UPLOAD_DIR, filename))
                file_path = filename

        summary = auto_summary(abstract)

        cur = db.execute(
            """INSERT INTO projects
               (title, abstract, student_name, supervisor, keywords, topic_area, department,
                institution, year, file_path, summary, builds_on_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (title, abstract, student_name, supervisor, keywords, topic_area, department,
             institution, int(year), file_path, summary, builds_on_id, datetime.now().isoformat()),
        )
        new_id = cur.lastrowid

        for row, score in flagged:
            db.execute(
                """INSERT INTO overlap_flags (project_id, similar_project_id, similarity_score)
                   VALUES (?, ?, ?)""",
                (new_id, row["id"], score),
            )

        db.commit()
        flash("Project submitted successfully.", "success")
        return redirect(url_for("project_detail", project_id=new_id))

    return render_template("submit.html", all_projects=all_projects, form={}, departments=DEPARTMENTS)


# ---------------------------------------------------------------------------
# Routes: supervisor / topic-gap dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
def dashboard():
    db = get_db()

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

    flagged_pairs = db.execute(
        """SELECT of.similarity_score, p1.id as pid1, p1.title as t1,
                  p2.id as pid2, p2.title as t2
           FROM overlap_flags of
           JOIN projects p1 ON of.project_id = p1.id
           JOIN projects p2 ON of.similar_project_id = p2.id
           ORDER BY of.similarity_score DESC LIMIT 10"""
    ).fetchall()

    return render_template(
        "dashboard.html",
        supervisor_load=supervisor_load,
        department_counts=department_counts,
        topic_counts=topic_counts,
        year_counts=year_counts,
        total_projects=total_projects,
        flagged_pairs=flagged_pairs,
    )


# ---------------------------------------------------------------------------
# JSON API (useful for the presentation demo / future frontend swap)
# ---------------------------------------------------------------------------

@app.route("/api/search")
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
        }
        for r, s in results
    ])


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)

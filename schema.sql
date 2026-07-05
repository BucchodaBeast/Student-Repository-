-- Launchpad schema — Supabase/Postgres
-- This is the same DDL that app.py's init_db() runs automatically on
-- startup (it's idempotent — IF NOT EXISTS everywhere), so you normally
-- don't need to run this by hand. It's provided separately in case you'd
-- rather set the schema up directly in the Supabase SQL editor first,
-- or want to inspect/version it outside the app.

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

-- Per-admin votes on pending projects. Two distinct admins approving moves
-- a project live; two distinct admins rejecting turns it away. UNIQUE keeps
-- each admin to exactly one live vote per project — re-voting overwrites it.
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

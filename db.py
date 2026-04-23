import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "bayes_planner.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    goal TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    parent_id INTEGER,
    prior REAL NOT NULL DEFAULT 0.5,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (parent_id) REFERENCES nodes(id),
    UNIQUE(project_id, code)
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL,
    result TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


@contextmanager
def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def now():
    return datetime.utcnow().isoformat(timespec="seconds")


def set_active_project(conn, project_id):
    conn.execute(
        "INSERT INTO state(key, value) VALUES('active_project', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(project_id),),
    )


def get_active_project(conn):
    row = conn.execute(
        "SELECT value FROM state WHERE key='active_project'"
    ).fetchone()
    if not row:
        return None
    pid = int(row["value"])
    return conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()


def create_project(conn, name, goal):
    conn.execute(
        "INSERT INTO projects(name, goal, created_at) VALUES(?, ?, ?)",
        (name, goal, now()),
    )
    row = conn.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()
    set_active_project(conn, row["id"])
    return row


def find_project(conn, name):
    return conn.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()


def list_projects(conn):
    return conn.execute(
        "SELECT * FROM projects ORDER BY created_at"
    ).fetchall()


def next_code(conn, project_id):
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM nodes WHERE project_id=?", (project_id,)
    ).fetchone()
    return f"jim{row['c']}"


def add_node(conn, project_id, name, description, parent_code, prior):
    parent_id = None
    if parent_code:
        p = conn.execute(
            "SELECT id FROM nodes WHERE project_id=? AND code=?",
            (project_id, parent_code),
        ).fetchone()
        if not p:
            raise ValueError(f"parent node '{parent_code}' not found")
        parent_id = p["id"]
    code = next_code(conn, project_id)
    # Beta prior centered on prior with weight 2 (lightly informative).
    alpha = max(0.5, 2 * prior)
    beta = max(0.5, 2 * (1 - prior))
    conn.execute(
        "INSERT INTO nodes(project_id, code, name, description, parent_id, "
        "prior, alpha, beta, status, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (project_id, code, name, description, parent_id, prior, alpha, beta,
         "pending", now()),
    )
    return conn.execute(
        "SELECT * FROM nodes WHERE project_id=? AND code=?",
        (project_id, code),
    ).fetchone()


def get_node(conn, project_id, code):
    return conn.execute(
        "SELECT * FROM nodes WHERE project_id=? AND code=?",
        (project_id, code),
    ).fetchone()


def list_nodes(conn, project_id):
    return conn.execute(
        "SELECT * FROM nodes WHERE project_id=? ORDER BY id",
        (project_id,),
    ).fetchall()


def list_children(conn, node_id):
    return conn.execute(
        "SELECT * FROM nodes WHERE parent_id=? ORDER BY id", (node_id,)
    ).fetchall()


def record_observation(conn, node_id, result, notes):
    conn.execute(
        "INSERT INTO observations(node_id, result, notes, created_at) "
        "VALUES(?,?,?,?)",
        (node_id, result, notes, now()),
    )


def update_node_belief(conn, node_id, alpha, beta, status):
    conn.execute(
        "UPDATE nodes SET alpha=?, beta=?, status=? WHERE id=?",
        (alpha, beta, status, node_id),
    )


def list_observations(conn, node_id):
    return conn.execute(
        "SELECT * FROM observations WHERE node_id=? ORDER BY id",
        (node_id,),
    ).fetchall()


def rollback_subtree(conn, project_id, code):
    """Mark the node and all descendants as 'obsolete' (kept for history)."""
    root = get_node(conn, project_id, code)
    if not root:
        raise ValueError(f"node '{code}' not found")
    to_visit = [root["id"]]
    touched = []
    while to_visit:
        nid = to_visit.pop()
        touched.append(nid)
        for c in list_children(conn, nid):
            to_visit.append(c["id"])
    conn.executemany(
        "UPDATE nodes SET status='obsolete' WHERE id=?",
        [(i,) for i in touched],
    )
    return touched

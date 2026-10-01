"""
Lightweight global usage metrics store.

Every tool on the site (JD Bias Reducer, Fair Hiring Index, Hiring AI
comparison) logs a real event here after a successful run, so the site can
report honest aggregate numbers across every visitor — not just the
visitor's own browser (that's what frontend/src/shared/fhiHistory.js is
for; this is the global counterpart).

Storage: a single SQLite file, no extra service required. Set
METRICS_DB_PATH to point at a persistent location — on Railway, attach a
Volume (Settings -> Volumes) and set METRICS_DB_PATH to a path inside it
(e.g. /data/metrics.db), otherwise counts reset whenever the service
redeploys, since Railway's default filesystem isn't persisted across
deploys.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "METRICS_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metrics.db"),
)

EVENT_TYPES = {"jd_analyzed", "fhi_submitted", "hiring_ai_compared"}
USER_TYPES = {"company", "individual"}


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_type TEXT NOT NULL CHECK(user_type IN ('company','individual')),
                company_name TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")


def register_user(user_id: str, user_type: str, company_name: str | None) -> None:
    if user_type not in USER_TYPES:
        raise ValueError("user_type must be 'company' or 'individual'")
    now = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        conn.execute("""
            INSERT INTO users (user_id, user_type, company_name, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                user_type=excluded.user_type,
                company_name=excluded.company_name
        """, (user_id, user_type, company_name, now))


def log_event(user_id: str, event_type: str, payload: dict) -> None:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event_type: {event_type}")
    now = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        # Attribute every event to *some* user even if the client never
        # finished registering (e.g. dismissed the company/individual
        # prompt) — default to an anonymous individual so usage still
        # counts toward totals instead of silently vanishing.
        conn.execute("""
            INSERT INTO users (user_id, user_type, company_name, created_at)
            VALUES (?, 'individual', NULL, ?)
            ON CONFLICT(user_id) DO NOTHING
        """, (user_id, now))
        conn.execute("""
            INSERT INTO events (user_id, event_type, payload, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, event_type, json.dumps(payload), now))


def summary() -> dict:
    with _db() as conn:
        total_users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
        companies = conn.execute(
            "SELECT COUNT(*) c FROM users WHERE user_type='company'").fetchone()["c"]
        individuals = conn.execute(
            "SELECT COUNT(*) c FROM users WHERE user_type='individual'").fetchone()["c"]

        jd_rows = conn.execute(
            "SELECT payload FROM events WHERE event_type='jd_analyzed'").fetchall()
        improvements = []
        for r in jd_rows:
            p = json.loads(r["payload"] or "{}")
            if isinstance(p.get("percent_improved"), (int, float)):
                improvements.append(p["percent_improved"])
        avg_improvement = round(sum(improvements) / len(improvements), 1) if improvements else 0.0

        fhi_rows = conn.execute(
            "SELECT payload FROM events WHERE event_type='fhi_submitted'").fetchall()
        fhi_values, total_people_covered = [], 0
        for r in fhi_rows:
            p = json.loads(r["payload"] or "{}")
            if isinstance(p.get("fhi"), (int, float)):
                fhi_values.append(p["fhi"])
            total_people_covered += p.get("team_size") or 0
        avg_fhi = round(sum(fhi_values) / len(fhi_values), 1) if fhi_values else None

        hiring_rows = conn.execute(
            "SELECT payload FROM events WHERE event_type='hiring_ai_compared'").fetchall()
        deltas = []
        for r in hiring_rows:
            p = json.loads(r["payload"] or "{}")
            if isinstance(p.get("score_delta"), (int, float)):
                deltas.append(p["score_delta"])
        avg_delta = round(sum(deltas) / len(deltas), 2) if deltas else None

        return {
            "jobDescriptions": {
                "totalAnalyzed": len(jd_rows),
                "avgPercentImproved": avg_improvement,
            },
            "fairHiringIndex": {
                "totalSubmissions": len(fhi_rows),
                "avgFhi": avg_fhi,
                "totalPeopleCovered": total_people_covered,
            },
            "hiringAi": {
                "totalComparisons": len(hiring_rows),
                "avgScoreDelta": avg_delta,
            },
            "users": {
                "total": total_users,
                "companies": companies,
                "individuals": individuals,
            },
        }

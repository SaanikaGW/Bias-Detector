"""
Testimonials / feedback store.

Lives in the same SQLite file as the usage metrics (metrics_store.DB_PATH),
so it persists on the same Railway volume. Submissions start as "pending"
and only appear on the public Testimonials page once approved.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import metrics_store

USER_TYPES = {"individual", "company"}
MAX_PER_DAY = 3  # per anonymous browser id, to blunt spam


def init_db() -> None:
    with metrics_store._db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                user_type TEXT NOT NULL,
                display_name TEXT,
                role TEXT,
                rating INTEGER NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)")


def _row(r) -> dict:
    return {
        "id": r["id"],
        "user_type": r["user_type"],
        "display_name": r["display_name"],  # None => shown as Anonymous
        "role": r["role"],
        "rating": r["rating"],
        "message": r["message"],
        "status": r["status"],
        "created_at": r["created_at"],
    }


def add(user_id: str, user_type: str, display_name: str | None,
        role: str | None, rating: int, message: str) -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    with metrics_store._db() as conn:
        recent = conn.execute(
            "SELECT COUNT(*) c FROM feedback WHERE user_id=? AND created_at>?",
            (user_id, since)).fetchone()["c"]
        if recent >= MAX_PER_DAY:
            raise PermissionError("Too many submissions today. Please try again tomorrow.")
        cur = conn.execute("""
            INSERT INTO feedback (user_id, user_type, display_name, role, rating, message, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (user_id, user_type, display_name, role, rating, message,
              datetime.now(timezone.utc).isoformat()))
        return cur.lastrowid


def list_by_status(status: str) -> list[dict]:
    with metrics_store._db() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    return [_row(r) for r in rows]


def set_status(feedback_id: int, status: str) -> bool:
    with metrics_store._db() as conn:
        cur = conn.execute("UPDATE feedback SET status=? WHERE id=?", (status, feedback_id))
        return cur.rowcount > 0


def delete(feedback_id: int) -> bool:
    with metrics_store._db() as conn:
        cur = conn.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
        return cur.rowcount > 0

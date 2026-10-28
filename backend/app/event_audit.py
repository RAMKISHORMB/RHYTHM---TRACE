
"""
TRACE — Persistent Event Review Audit
MEDAITHON'26 | Cardiology PS-001

Append-only application-level SQLite event history.

Research prototype only.
No authentication or tamper-proof guarantees.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.event_api import load_demo_events


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "trace_event_audit.db"

router = APIRouter(
    prefix="/api/event-audit",
    tags=["TRACE Event Review Audit"],
)

ALLOWED_ACTIONS = {
    "REVIEWED",
    "NEEDS_FOLLOWUP",
}


class ReviewRequest(BaseModel):
    event_id: str
    action: str
    reviewer: str = Field(
        min_length=1,
        max_length=80,
    )
    note: str = Field(
        default="",
        max_length=1000,
    )


def connect_db():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    connection.execute("""
        CREATE TABLE IF NOT EXISTS event_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            action TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            note TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL
        )
    """)

    connection.commit()
    return connection


def ensure_known_event(event_id: str):
    try:
        events = load_demo_events()["events"]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Event evidence unavailable.",
        ) from exc

    if not any(
        event["event_id"] == event_id
        for event in events
    ):
        raise HTTPException(
            status_code=404,
            detail="Event not found.",
        )


@router.post("/review")
def record_review(request: ReviewRequest):
    ensure_known_event(request.event_id)

    action = request.action.strip().upper()

    if action not in ALLOWED_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail="Unsupported review action.",
        )

    reviewer = request.reviewer.strip()

    if not reviewer:
        raise HTTPException(
            status_code=422,
            detail="Reviewer cannot be blank.",
        )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with connect_db() as connection:
        cursor = connection.execute(
            """
            INSERT INTO event_audit (
                event_id,
                action,
                reviewer,
                note,
                timestamp_utc
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request.event_id,
                action,
                reviewer,
                request.note,
                timestamp,
            ),
        )

        audit_id = cursor.lastrowid

    return {
        "audit_id": audit_id,
        "event_id": request.event_id,
        "action": action,
        "reviewer": reviewer,
        "note": request.note,
        "timestamp_utc": timestamp,
        "saved": True,
        "research_only": True,
    }


@router.get("/{event_id}")
def get_event_audit(event_id: str):
    ensure_known_event(event_id)

    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                event_id,
                action,
                reviewer,
                note,
                timestamp_utc
            FROM event_audit
            WHERE event_id = ?
            ORDER BY id ASC
            """,
            (event_id,),
        ).fetchall()

    history = [dict(row) for row in rows]

    return {
        "event_id": event_id,
        "review_count": len(history),
        "current_review_status": (
            history[-1]["action"]
            if history
            else "PENDING"
        ),
        "history": history,
        "research_only": True,
    }

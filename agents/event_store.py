"""Persistent Event Store for PoMAS Multi-Agent System.

Stores all events (EEG windows, suspicion flags, prediction requests, risk scores,
and consolidated alerts) in an SQLite database for replay auditability and dashboard consumption.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import OUTPUT_DIR

DEFAULT_DB_PATH = OUTPUT_DIR / "pomas_event_store.db"


class EventStore:
    """SQLite-backed event store for the PoMAS agentic pipeline."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create events table with appropriate indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    patient_id TEXT NOT NULL,
                    recording_id TEXT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    risk_score REAL,
                    ground_truth INTEGER,
                    alert_tier TEXT,
                    details TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_patient ON events(patient_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_tier ON events(alert_tier)")
            conn.commit()

    def record_event(
        self,
        event_type: str,
        patient_id: str,
        timestamp: float,
        recording_id: Optional[str] = None,
        risk_score: Optional[float] = None,
        ground_truth: Optional[int] = None,
        alert_tier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> int:
        """Record a single event in the persistent store."""
        created_at = datetime.utcnow().isoformat()
        details_json = json.dumps(details or {})
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    event_id, patient_id, recording_id, timestamp, event_type,
                    risk_score, ground_truth, alert_tier, details, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id or "",
                    patient_id,
                    recording_id or "",
                    float(timestamp),
                    str(event_type),
                    float(risk_score) if risk_score is not None else None,
                    int(ground_truth) if ground_truth is not None else None,
                    str(alert_tier) if alert_tier is not None else None,
                    details_json,
                    created_at,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def record_batch(self, records: List[Dict[str, Any]]) -> None:
        """Efficiently record a batch of events."""
        if not records:
            return
        created_at = datetime.utcnow().isoformat()
        rows = [
            (
                r.get("event_id", ""),
                r.get("patient_id", "unknown"),
                r.get("recording_id", ""),
                float(r.get("timestamp", 0.0)),
                str(r.get("event_type", "unknown")),
                float(r["risk_score"]) if r.get("risk_score") is not None else None,
                int(r["ground_truth"]) if r.get("ground_truth") is not None else None,
                str(r["alert_tier"]) if r.get("alert_tier") is not None else None,
                json.dumps(r.get("details", {})),
                created_at,
            )
            for r in records
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO events (
                    event_id, patient_id, recording_id, timestamp, event_type,
                    risk_score, ground_truth, alert_tier, details, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def get_recent_events(
        self, patient_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve recent events, optionally filtered by patient."""
        query = "SELECT * FROM events"
        params: List[Any] = []
        if patient_id:
            query += " WHERE patient_id = ?"
            params.append(patient_id)
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_alerts(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all raised alert events."""
        query = "SELECT * FROM events WHERE alert_tier IS NOT NULL"
        params: List[Any] = []
        if patient_id:
            query += " AND patient_id = ?"
            params.append(patient_id)
        query += " ORDER BY timestamp ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_timeline(
        self,
        patient_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve chronological risk score timeline."""
        query = "SELECT timestamp, risk_score, ground_truth, alert_tier FROM events WHERE patient_id = ? AND risk_score IS NOT NULL"
        params: List[Any] = [patient_id]
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        query += " ORDER BY timestamp ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def clear(self, patient_id: Optional[str] = None) -> None:
        """Clear recorded events (for clean replay runs)."""
        with self._get_connection() as conn:
            if patient_id:
                conn.execute("DELETE FROM events WHERE patient_id = ?", (patient_id,))
            else:
                conn.execute("DELETE FROM events")
            conn.commit()

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from dmease.io import load_patients
from dmease.schemas import PatientRecord
from dmease.types import AnalysisResult


class PatientDB:
    """Lightweight PatientDB indexed by patient-ID, as described in the paper."""

    def __init__(self, records: dict[str, PatientRecord]):
        self._records = records

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PatientDB":
        return cls(load_patients(path))

    def get(self, patient_id: str) -> PatientRecord:
        try:
            return self._records[patient_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._records)[:10]) or "none"
            raise KeyError(f"Unknown patient_id={patient_id!r}; available examples: {known}") from exc

    def upsert(self, record: PatientRecord) -> None:
        self._records[record.patient_id] = record

    def list_ids(self) -> list[str]:
        return sorted(self._records)


SCHEMA = """
CREATE TABLE IF NOT EXISTS patient_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    complaint TEXT NOT NULL,
    primary_syndrome TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patient_records_name ON patient_records(patient_name);
CREATE INDEX IF NOT EXISTS idx_patient_records_created_at ON patient_records(created_at);
"""


class PatientDatabase:
    """Persistent SQLite store for interactive patient analysis records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, result: AnalysisResult) -> int:
        primary = result.syndromes[0].name if result.syndromes else None
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO patient_records(patient_name, created_at, complaint, primary_syndrome, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    result.patient.name,
                    result.created_at,
                    result.patient.complaint,
                    primary,
                    json.dumps(result.to_dict(), ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, patient_name, created_at, complaint, primary_syndrome "
                "FROM patient_records ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, record_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM patient_records WHERE id = ?", (record_id,)
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def delete(self, record_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
            return cursor.rowcount > 0


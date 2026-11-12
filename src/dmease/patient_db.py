from __future__ import annotations

from contextlib import closing

import json
import hashlib
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
        with closing(self._connect()) as connection, connection:
            connection.executescript(SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(patient_records)")}
            for name, definition in (("owner", "TEXT NOT NULL DEFAULT ''"),
                                     ("result_key", "TEXT"), ("parent_record_id", "INTEGER")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE patient_records ADD COLUMN {name} {definition}")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_record_request "
                               "ON patient_records(owner,result_key)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, result: AnalysisResult, owner: str = "", parent_record_id: int | None = None) -> int:
        primary = result.syndromes[0].name if result.syndromes else None
        payload = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)
        result_key = hashlib.sha256(payload.encode()).hexdigest()
        with closing(self._connect()) as connection, connection:
            if parent_record_id is not None and not connection.execute(
                "SELECT 1 FROM patient_records WHERE id=? AND owner=?", (parent_record_id, owner)
            ).fetchone():
                raise ValueError("原始记录不存在或不属于当前账号。")
            cursor = connection.execute(
                "INSERT OR IGNORE INTO patient_records(patient_name, created_at, complaint, primary_syndrome, "
                "payload_json,owner,result_key,parent_record_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.patient.name,
                    result.created_at,
                    result.patient.complaint,
                    primary,
                    payload, owner, result_key, parent_record_id,
                ),
            )
            row = connection.execute("SELECT id FROM patient_records WHERE owner=? AND result_key=?",
                                     (owner, result_key)).fetchone()
            return int(row[0])

    def list(self, limit: int = 100, *, owner: str | None = None, query: str = "",
             syndrome: str = "") -> list[dict[str, Any]]:
        clauses, args = [], []
        if owner is not None:
            clauses.append("owner=?"); args.append(owner)
        if query.strip():
            clauses.append("(instr(patient_name,?)>0 OR instr(complaint,?)>0)")
            args.extend([query.strip(), query.strip()])
        if syndrome:
            clauses.append("primary_syndrome=?"); args.append(syndrome)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id, patient_name, created_at, complaint, primary_syndrome, parent_record_id "
                "FROM patient_records" + where + " ORDER BY id DESC LIMIT ?",
                (*args, max(1, min(limit, 10000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self, owner: str) -> int:
        with closing(self._connect()) as connection, connection:
            return int(connection.execute("SELECT COUNT(*) FROM patient_records WHERE owner=?", (owner,)).fetchone()[0])

    def get(self, record_id: int, *, owner: str | None = None) -> dict[str, Any] | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT payload_json FROM patient_records WHERE id = ?" + (" AND owner=?" if owner is not None else ""),
                (record_id, owner) if owner is not None else (record_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def delete(self, record_id: int, *, owner: str | None = None) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("DELETE FROM patient_records WHERE id = ?" +
                                        (" AND owner=?" if owner is not None else ""),
                                        (record_id, owner) if owner is not None else (record_id,))
            return cursor.rowcount > 0


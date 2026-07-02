from __future__ import annotations

from pathlib import Path

from dmease.io import load_patients
from dmease.schemas import PatientRecord


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


from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import yaml

from dmease.schemas import PatientRecord, Triple


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    source = Path(path)
    if not source.exists():
        return records
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_patients(path: str | Path) -> dict[str, PatientRecord]:
    return {record.patient_id: record for record in (PatientRecord.from_dict(row) for row in read_jsonl(path))}


def load_triples(path: str | Path) -> list[Triple]:
    return [Triple.from_dict(row) for row in read_jsonl(path)]


def load_herb_target_affinity(path: str | Path) -> dict[str, dict[str, float]]:
    source = Path(path)
    if not source.exists():
        return {}
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return {}
        target_columns = [name for name in reader.fieldnames if name != "herb"]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            herb = str(row["herb"]).strip()
            matrix[herb] = {
                target: float(row[target] or 0.0)
                for target in target_columns
                if row.get(target) not in (None, "")
            }
        return matrix


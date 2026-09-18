from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ParsedSymptom:
    canonical: str
    original: str
    present: bool = True
    severity: float = 1.0
    evidence: str = ""


@dataclass(slots=True)
class PatientProfile:
    name: str = "匿名患者"
    age: int = 50
    sex: str = "未说明"
    constitution: str = "平和质"
    complaint: str = ""
    allergies: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    pregnant: bool = False
    medications: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class SyndromeScore:
    name: str
    score: float
    confidence: float
    matched_symptoms: list[str]
    missing_key_symptoms: list[str]
    rationale: str


@dataclass(slots=True)
class HerbRecommendation:
    name: str
    score: float
    rank: int
    actions: list[str]
    matched_syndromes: list[str]
    evidence_paths: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisResult:
    patient: PatientProfile
    parsed_symptoms: list[ParsedSymptom]
    syndromes: list[SyndromeScore]
    recommendations: list[HerbRecommendation]
    excluded_herbs: dict[str, list[str]]
    safety_notes: list[str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

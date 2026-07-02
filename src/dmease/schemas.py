from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Triple:
    subject: str
    predicate: str
    object: str
    source: str = "unspecified source"
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Triple":
        return cls(
            subject=str(row["subject"]).strip(),
            predicate=str(row["predicate"]).strip(),
            object=str(row["object"]).strip(),
            source=str(row.get("source", "unspecified source")).strip(),
            confidence=float(row.get("confidence", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class PatientRecord:
    patient_id: str
    symptoms: dict[str, float]
    constitution: dict[str, float] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    previous_herbs: list[str] = field(default_factory=list)
    constraints: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PatientRecord":
        return cls(
            patient_id=str(row["patient_id"]),
            symptoms={str(k): float(v) for k, v in row.get("symptoms", {}).items()},
            constitution={str(k): float(v) for k, v in row.get("constitution", {}).items()},
            history=[str(item) for item in row.get("history", [])],
            targets=[str(item) for item in row.get("targets", [])],
            previous_herbs=[str(item) for item in row.get("previous_herbs", [])],
            constraints={
                str(k): [str(item) for item in v] for k, v in row.get("constraints", {}).items()
            },
        )


@dataclass(slots=True)
class SyndromePrediction:
    syndrome: str
    score: float
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HerbCandidate:
    herb: str
    score: float
    covered_targets: list[str]
    matched_symptoms: list[str]
    contraindications: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PrescriptionStep:
    step: int
    herb: str
    marginal_relief: float
    residual_symptoms: dict[str, float]
    trace: dict[str, float]
    covered_targets: list[str]
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InferenceResult:
    patient_id: str
    syndrome: SyndromePrediction
    targets: list[str]
    ranked_candidates: list[HerbCandidate]
    prescription: list[PrescriptionStep]
    target_consistency_score: float
    notes: list[str] = field(default_factory=list)

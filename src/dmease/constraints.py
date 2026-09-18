from __future__ import annotations

from dmease.schemas import PatientRecord
from dmease.types import PatientProfile
from collections import defaultdict


class CompatibilityConstraints:
    """Classical and patient-specific hard constraints."""

    def __init__(self, forbidden_pairs: set[tuple[str, str]]):
        self.forbidden_pairs = forbidden_pairs

    def reasons(self, herb: str, selected_herbs: list[str], patient: PatientRecord) -> list[str]:
        reasons: list[str] = []
        for chosen in selected_herbs:
            if (herb, chosen) in self.forbidden_pairs:
                reasons.append(f"{herb} 与 {chosen} 存在配伍禁忌")

        allergies = set(patient.constraints.get("allergy", []))
        pregnancy_forbidden = set(patient.constraints.get("pregnancy_forbidden", []))
        if herb in allergies:
            reasons.append(f"{herb} 命中患者过敏约束")
        if herb in pregnancy_forbidden:
            reasons.append(f"{herb} 命中妊娠/特殊人群禁忌约束")
        return reasons

    def is_allowed(self, herb: str, selected_herbs: list[str], patient: PatientRecord) -> bool:
        return not self.reasons(herb, selected_herbs, patient)


class SafetyConstraintEngine:
    """Applies patient-specific exclusions and produces auditable reasons."""

    def evaluate(self, herb: dict, patient: PatientProfile) -> list[str]:
        reasons: list[str] = []
        contraindications = set(herb.get("contraindications", []))
        if patient.pregnant and "妊娠" in contraindications:
            reasons.append("妊娠期禁用或慎用")
        for condition in patient.conditions:
            if condition in contraindications:
                reasons.append(f"合并情况“{condition}”与知识库禁忌相符")
        for allergy in patient.allergies:
            if allergy and (allergy in herb["name"] or allergy in herb.get("aliases", [])):
                reasons.append(f"患者记录了“{allergy}”过敏")
        interactions = herb.get("interactions", {})
        for medication in patient.medications:
            for key, note in interactions.items():
                if key in medication or medication in key:
                    reasons.append(f"与当前用药“{medication}”可能相互作用：{note}")
        return sorted(set(reasons))

    def incompatibilities(self, herbs: list[dict]) -> dict[str, list[str]]:
        selected = {item["name"] for item in herbs}
        conflicts: dict[str, list[str]] = defaultdict(list)
        for herb in herbs:
            for other in herb.get("incompatible", []):
                if other in selected:
                    conflicts[herb["name"]].append(f"与{other}存在配伍禁忌")
                    conflicts[other].append(f"与{herb['name']}存在配伍禁忌")
        return dict(conflicts)


DEFAULT_SAFETY_NOTES = [
    "结果用于科研与医生决策支持，不构成独立诊断、处方或用药建议。",
    "实际应用必须由具有资质的中医师完成四诊合参，并核对剂量、炮制、配伍与药物相互作用。",
    "出现意识改变、酮症相关表现、严重低血糖或血糖持续异常时，应立即按正规医疗流程处置。",
]


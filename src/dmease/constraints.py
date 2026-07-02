from __future__ import annotations

from dmease.schemas import PatientRecord


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


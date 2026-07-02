from __future__ import annotations


def target_consistency_score(symptom_targets: list[str], herb_targets: list[str]) -> float:
    symptom_set = set(symptom_targets)
    herb_set = set(herb_targets)
    if not symptom_set and not herb_set:
        return 0.0
    return len(symptom_set & herb_set) / len(symptom_set | herb_set)


def residual_norm(symptoms: dict[str, float]) -> float:
    return sum(abs(value) for value in symptoms.values())


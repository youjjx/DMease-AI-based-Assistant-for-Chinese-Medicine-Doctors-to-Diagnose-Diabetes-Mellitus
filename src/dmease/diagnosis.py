from __future__ import annotations

from dmease.knowledge_graph import TCMKnowledgeGraph
from dmease.schemas import SyndromePrediction
from dmease.knowledge_graph import KnowledgeGraph
from dmease.types import ParsedSymptom, PatientProfile, SyndromeScore
import math


class SyndromeDiagnosisModule:
    """Symptom-to-zheng diagnosis through graph voting."""

    def __init__(self, graph: TCMKnowledgeGraph):
        self.graph = graph

    def predict(self, symptoms: dict[str, float]) -> SyndromePrediction:
        syndrome_scores = self.graph.syndromes_for_symptoms(list(symptoms))
        if not syndrome_scores:
            return SyndromePrediction(
                syndrome="未匹配证候",
                score=0.0,
                evidence=["当前症状组合未触发已配置的证候规则，请结合临床信息复核。"],
            )

        weighted: dict[str, tuple[float, list[str]]] = {}
        for syndrome, (score, evidence) in syndrome_scores.items():
            symptom_weight = sum(
                severity
                for symptom, severity in symptoms.items()
                if any(item.startswith(f"{symptom} ->") for item in evidence)
            )
            weighted[syndrome] = (score * max(symptom_weight, 1.0), evidence)

        syndrome, (score, evidence) = max(weighted.items(), key=lambda item: item[1][0])
        normalizer = sum(value[0] for value in weighted.values()) or 1.0
        return SyndromePrediction(syndrome=syndrome, score=score / normalizer, evidence=evidence)


class SyndromeReasoner:
    """Interpretable weighted evidence reasoner over the structured graph."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def infer(
        self,
        symptoms: list[ParsedSymptom],
        patient: PatientProfile,
        top_k: int = 3,
    ) -> list[SyndromeScore]:
        observed = {item.canonical: item.severity for item in symptoms if item.present}
        denied = {item.canonical for item in symptoms if not item.present}
        raw: list[tuple[dict, float, list[str], list[str]]] = []
        for syndrome in self.graph.syndromes:
            expected = syndrome["symptoms"]
            numerator = sum(float(weight) * observed.get(name, 0.0) for name, weight in expected.items())
            denominator = max(sum(float(weight) for weight in expected.values()), 1e-6)
            contradiction = sum(float(expected[name]) for name in denied if name in expected) / denominator
            constitution_bonus = 0.12 if patient.constitution in syndrome.get("constitutions", []) else 0.0
            score = max(0.0, numerator / denominator + constitution_bonus - 0.35 * contradiction)
            matched = sorted((name for name in observed if name in expected), key=lambda name: expected[name], reverse=True)
            missing = sorted((name for name in expected if name not in observed), key=lambda name: expected[name], reverse=True)[:3]
            raw.append((syndrome, score, matched, missing))

        logits = [item[1] * 5.0 for item in raw]
        max_logit = max(logits, default=0.0)
        exps = [math.exp(value - max_logit) for value in logits]
        normalizer = sum(exps) or 1.0
        result = []
        for (syndrome, score, matched, missing), exp_value in zip(raw, exps):
            result.append(
                SyndromeScore(
                    name=syndrome["name"],
                    score=round(score, 4),
                    confidence=round(exp_value / normalizer, 4),
                    matched_symptoms=matched,
                    missing_key_symptoms=missing,
                    rationale=syndrome.get("description", ""),
                )
            )
        return sorted(result, key=lambda item: item.score, reverse=True)[:top_k]

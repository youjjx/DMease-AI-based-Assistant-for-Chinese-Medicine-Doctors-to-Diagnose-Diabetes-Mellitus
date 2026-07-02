from __future__ import annotations

from dmease.knowledge_graph import TCMKnowledgeGraph
from dmease.schemas import SyndromePrediction


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

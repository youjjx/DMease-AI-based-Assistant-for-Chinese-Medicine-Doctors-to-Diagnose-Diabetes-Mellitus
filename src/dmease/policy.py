from __future__ import annotations

from collections import defaultdict
from math import exp

from dmease.constraints import CompatibilityConstraints
from dmease.knowledge_graph import HERB_AFFECTS_TARGET, SYNDROME_TREATED_BY, TCMKnowledgeGraph
from dmease.schemas import HerbCandidate, PatientRecord


class SequentialHerbPolicy:
    """Greedy single-herb policy used until trained KAN weights are available."""

    def __init__(
        self,
        graph: TCMKnowledgeGraph,
        herb_target_affinity: dict[str, dict[str, float]],
        constraints: CompatibilityConstraints,
        contraindication_penalty: float = 0.35,
        affinity_threshold: float = 0.5,
    ):
        self.graph = graph
        self.herb_target_affinity = herb_target_affinity
        self.constraints = constraints
        self.contraindication_penalty = contraindication_penalty
        self.affinity_threshold = affinity_threshold

    def rank_candidates(
        self,
        patient: PatientRecord,
        symptoms: dict[str, float],
        syndrome: str,
        targets: list[str],
        selected_herbs: list[str] | None = None,
    ) -> list[HerbCandidate]:
        selected_herbs = selected_herbs or []
        herbs = self._candidate_herbs(syndrome, targets)
        candidates = [
            self._score_herb(herb, patient, symptoms, targets, selected_herbs)
            for herb in herbs
            if herb not in selected_herbs and herb not in patient.previous_herbs
        ]
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _candidate_herbs(self, syndrome: str, targets: list[str]) -> list[str]:
        candidates: set[str] = set()
        for triple in self.graph.herbs_for_syndrome(syndrome):
            candidates.add(triple.object)

        target_set = set(targets)
        for herb, affinity in self.herb_target_affinity.items():
            if target_set.intersection(affinity):
                candidates.add(herb)

        for triple in self.graph.triples:
            if triple.predicate == HERB_AFFECTS_TARGET and triple.object in target_set:
                candidates.add(triple.subject)
        return sorted(candidates)

    def _score_herb(
        self,
        herb: str,
        patient: PatientRecord,
        symptoms: dict[str, float],
        targets: list[str],
        selected_herbs: list[str],
    ) -> HerbCandidate:
        affinity = self.herb_target_affinity.get(herb, {})
        target_set = set(targets)
        graph_edges = self.graph.herb_target_edges(herb)
        graph_confidence = {
            edge.object: max(edge.confidence, affinity.get(edge.object, 0.0))
            for edge in graph_edges
            if edge.object in target_set
        }
        covered_targets = sorted(
            target
            for target in target_set
            if max(affinity.get(target, 0.0), graph_confidence.get(target, 0.0))
            >= self.affinity_threshold
        )
        for edge in graph_edges:
            if edge.object in target_set and edge.object not in covered_targets:
                covered_targets.append(edge.object)

        relief = sum(max(affinity.get(target, 0.0), graph_confidence.get(target, 0.0)) for target in covered_targets)
        symptom_factor = sum(symptoms.values()) / max(len(symptoms), 1)
        reward = symptom_factor * (1.0 - exp(-relief))

        contraindications = self.constraints.reasons(herb, selected_herbs, patient)
        if contraindications:
            reward -= self.contraindication_penalty * len(contraindications)

        syndrome_bonus = 0.0
        provenance: list[str] = []
        for triple in self.graph.triples:
            if triple.predicate == SYNDROME_TREATED_BY and triple.object == herb:
                syndrome_bonus += 0.05 * triple.confidence
                provenance.append(triple.source)
            if triple.subject == herb and triple.object in covered_targets:
                provenance.append(triple.source)
        if affinity and covered_targets:
            provenance.append("Configured herb-target affinity matrix")

        matched_symptoms = self._matched_symptoms(targets=covered_targets)
        score = reward + syndrome_bonus
        return HerbCandidate(
            herb=herb,
            score=round(score, 6),
            covered_targets=covered_targets,
            matched_symptoms=matched_symptoms,
            contraindications=contraindications,
            provenance=sorted(set(provenance)),
        )

    def symptom_trace(self, herb: str, symptoms: dict[str, float], targets: list[str]) -> dict[str, float]:
        affinity = self.herb_target_affinity.get(herb, {})
        graph_confidence = {
            edge.object: max(edge.confidence, affinity.get(edge.object, 0.0))
            for edge in self.graph.herb_target_edges(herb)
        }
        trace: dict[str, float] = {}
        for symptom in symptoms:
            related = self.graph.targets_for_symptoms([symptom])
            numerator = sum(
                max(affinity.get(target, 0.0), graph_confidence.get(target, 0.0))
                for target in related
                if target in targets
                and max(affinity.get(target, 0.0), graph_confidence.get(target, 0.0))
                >= self.affinity_threshold
            )
            trace[symptom] = round(numerator * symptoms[symptom], 6)
        return trace

    def _matched_symptoms(self, targets: list[str]) -> list[str]:
        reverse: dict[str, set[str]] = defaultdict(set)
        for triple in self.graph.triples:
            if triple.predicate in {"associated with", "associatedWith"} and triple.object in targets:
                reverse[triple.object].add(triple.subject)
        symptoms = sorted({symptom for values in reverse.values() for symptom in values})
        return symptoms

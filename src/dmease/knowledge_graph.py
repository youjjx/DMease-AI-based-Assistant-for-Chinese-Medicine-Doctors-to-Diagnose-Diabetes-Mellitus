from __future__ import annotations

from collections import Counter, defaultdict
from math import log

import networkx as nx

from dmease.schemas import Triple


SYMPTOM_TARGET = "associated with"
SYMPTOM_INDICATES = "indicates"
SYNDROME_TREATED_BY = "treatedBy"
HERB_AFFECTS_TARGET = "affect by"
HERB_CONTRAINDICATED = "contraindicatedWith"
COMPOUND_TARGETS_GENE = "targets"


class TCMKnowledgeGraph:
    """Symptom-target-herb graph with provenance attached to edges."""

    def __init__(self, triples: list[Triple]):
        self.triples = triples
        self.graph = nx.MultiDiGraph()
        self._build()

    def _build(self) -> None:
        for triple in self.triples:
            self.graph.add_edge(
                triple.subject,
                triple.object,
                key=triple.predicate,
                predicate=triple.predicate,
                source=triple.source,
                confidence=triple.confidence,
            )

    def targets_for_symptoms(self, symptoms: list[str], pmi_threshold: float = 0.0) -> list[str]:
        scores = self.symptom_target_scores(symptoms)
        return [target for target, score in scores.items() if score > pmi_threshold]

    def symptom_target_scores(self, symptoms: list[str]) -> dict[str, float]:
        """Frequency-weighted PMI approximation from graph co-occurrence."""

        symptom_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()

        for triple in self.triples:
            if triple.predicate not in {SYMPTOM_TARGET, "associatedWith"}:
                continue
            symptom_counts[triple.subject] += 1
            target_counts[triple.object] += 1
            pair_counts[(triple.subject, triple.object)] += 1

        total_pairs = sum(pair_counts.values()) or 1
        scores: dict[str, float] = defaultdict(float)
        for symptom in symptoms:
            for (symptom_name, target), pair_count in pair_counts.items():
                if symptom_name != symptom:
                    continue
                p_pair = pair_count / total_pairs
                p_symptom = symptom_counts[symptom_name] / total_pairs
                p_target = target_counts[target] / total_pairs
                pmi = log((p_pair + 1e-9) / ((p_symptom * p_target) + 1e-9))
                tf_idf_like = 1.0 / (1.0 + log(1.0 + target_counts[target]))
                scores[target] += max(0.0, pmi) * tf_idf_like
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def syndromes_for_symptoms(self, symptoms: list[str]) -> dict[str, tuple[float, list[str]]]:
        votes: dict[str, float] = defaultdict(float)
        evidence: dict[str, list[str]] = defaultdict(list)
        for triple in self.triples:
            if triple.predicate != SYMPTOM_INDICATES or triple.subject not in symptoms:
                continue
            votes[triple.object] += triple.confidence
            evidence[triple.object].append(f"{triple.subject} -> {triple.object} ({triple.source})")
        return {key: (votes[key], evidence[key]) for key in votes}

    def herbs_for_syndrome(self, syndrome: str) -> list[Triple]:
        return [
            triple
            for triple in self.triples
            if triple.predicate == SYNDROME_TREATED_BY and triple.subject == syndrome
        ]

    def herb_target_edges(self, herb: str) -> list[Triple]:
        return [
            triple
            for triple in self.triples
            if triple.subject == herb and triple.predicate in {HERB_AFFECTS_TARGET, COMPOUND_TARGETS_GENE}
        ]

    def contraindicated_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for triple in self.triples:
            if triple.predicate == HERB_CONTRAINDICATED:
                pairs.add((triple.subject, triple.object))
                pairs.add((triple.object, triple.subject))
        return pairs

    def provenance_for_edge(self, subject: str, predicate: str, object_: str) -> list[str]:
        return [
            triple.source
            for triple in self.triples
            if triple.subject == subject and triple.predicate == predicate and triple.object == object_
        ]


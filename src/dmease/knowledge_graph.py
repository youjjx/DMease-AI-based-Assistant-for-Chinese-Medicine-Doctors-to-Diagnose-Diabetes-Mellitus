from __future__ import annotations

from collections import Counter, defaultdict
import json
from math import log
from pathlib import Path
from typing import Iterable

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


class KnowledgeGraph:
    """JSON-backed clinical research graph used by the interactive workflow."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.metadata = payload.get("metadata", {})
        self.symptoms = payload["symptoms"]
        self.syndromes = payload["syndromes"]
        self.herbs = payload["herbs"]
        self.compounds = payload.get("compounds", [])
        self.targets = payload.get("targets", [])
        self.pathways = payload.get("pathways", [])
        self.mechanism_edges = payload.get("mechanism_edges", [])
        self._symptom = {item["name"]: item for item in self.symptoms}
        self._syndrome = {item["name"]: item for item in self.syndromes}
        self._herb = {item["name"]: item for item in self.herbs}
        self._herbs_by_syndrome: dict[str, list[dict]] = defaultdict(list)
        for herb in self.herbs:
            for syndrome, weight in herb.get("syndromes", {}).items():
                self._herbs_by_syndrome[syndrome].append({"herb": herb, "weight": float(weight)})
        self._validate()

    def _validate(self) -> None:
        for label, values in (
            ("symptom", [item["name"] for item in self.symptoms]),
            ("syndrome", [item["name"] for item in self.syndromes]),
            ("herb", [item["name"] for item in self.herbs]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label} name in knowledge graph")
        for syndrome in self.syndromes:
            unknown = set(syndrome.get("symptoms", {})) - set(self._symptom)
            if unknown:
                raise ValueError(f"Unknown symptoms in {syndrome['name']}: {sorted(unknown)}")
        for herb in self.herbs:
            unknown = set(herb.get("syndromes", {})) - set(self._syndrome)
            if unknown:
                raise ValueError(f"Unknown syndromes in {herb['name']}: {sorted(unknown)}")
        known_nodes = set(self._herb) | set(self.compounds) | set(self.targets) | set(self.pathways)
        for edge in self.mechanism_edges:
            if edge["source"] not in known_nodes or edge["target"] not in known_nodes:
                raise ValueError(f"Unknown node in mechanism edge: {edge}")

    @property
    def symptom_names(self) -> list[str]:
        return sorted(self._symptom)

    @property
    def syndrome_names(self) -> list[str]:
        return sorted(self._syndrome)

    @property
    def herb_names(self) -> list[str]:
        return sorted(self._herb)

    def syndrome(self, name: str) -> dict:
        return self._syndrome[name]

    def herb(self, name: str) -> dict:
        return self._herb[name]

    def herbs_for_syndrome(self, name: str) -> list[dict]:
        return sorted(self._herbs_by_syndrome.get(name, []), key=lambda item: item["weight"], reverse=True)

    def paths(self, symptoms: Iterable[str], syndrome: str, herb: str) -> list[str]:
        recognized = self._syndrome.get(syndrome, {}).get("symptoms", {})
        return [
            f"{symptom} → 支持证候 → {syndrome} → 关联用药 → {herb}"
            for symptom in symptoms
            if symptom in recognized
        ][:3]

    def mechanism_paths(self, herb: str) -> list[str]:
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in self.mechanism_edges:
            adjacency[edge["source"]].append((edge["relation"], edge["target"]))
        paths = []
        for relation_1, compound in adjacency.get(herb, []):
            for relation_2, target in adjacency.get(compound, []):
                downstream = adjacency.get(target, [])
                if downstream:
                    for relation_3, pathway in downstream:
                        paths.append(
                            f"{herb} —{relation_1}→ {compound} —{relation_2}→ "
                            f"{target} —{relation_3}→ {pathway}"
                        )
                else:
                    paths.append(f"{herb} —{relation_1}→ {compound} —{relation_2}→ {target}")
        return paths

    def summary(self) -> dict[str, int | str]:
        relations = sum(len(item.get("symptoms", {})) for item in self.syndromes)
        relations += sum(len(item.get("syndromes", {})) for item in self.herbs)
        relations += len(self.mechanism_edges)
        return {
            "version": self.metadata.get("version", "unknown"),
            "symptoms": len(self.symptoms),
            "syndromes": len(self.syndromes),
            "herbs": len(self.herbs),
            "compounds": len(self.compounds),
            "targets": len(self.targets),
            "pathways": len(self.pathways),
            "relations": relations,
        }


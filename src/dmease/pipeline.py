from __future__ import annotations

from pathlib import Path

from dmease.constraints import CompatibilityConstraints
from dmease.diagnosis import SyndromeDiagnosisModule
from dmease.io import load_config, load_herb_target_affinity, load_triples
from dmease.knowledge_graph import TCMKnowledgeGraph
from dmease.metrics import residual_norm, target_consistency_score
from dmease.patient_db import PatientDB
from dmease.policy import SequentialHerbPolicy
from dmease.schemas import InferenceResult, PrescriptionStep
from dmease.semantic_parser import SemanticParser


class DMeasePipeline:
    def __init__(
        self,
        patient_db: PatientDB,
        graph: TCMKnowledgeGraph,
        herb_target_affinity: dict[str, dict[str, float]],
        max_herbs: int = 7,
        residual_threshold: float = 0.08,
        pmi_threshold: float = 0.0,
        contraindication_penalty: float = 0.35,
    ):
        self.patient_db = patient_db
        self.graph = graph
        self.parser = SemanticParser()
        self.diagnoser = SyndromeDiagnosisModule(graph)
        constraints = CompatibilityConstraints(graph.contraindicated_pairs())
        self.policy = SequentialHerbPolicy(
            graph=graph,
            herb_target_affinity=herb_target_affinity,
            constraints=constraints,
            contraindication_penalty=contraindication_penalty,
        )
        self.max_herbs = max_herbs
        self.residual_threshold = residual_threshold
        self.pmi_threshold = pmi_threshold

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DMeasePipeline":
        config_path = Path(config_path)
        cfg = load_config(config_path)
        base_dir = config_path.parent.parent

        def resolve(path: str) -> Path:
            candidate = Path(path)
            return candidate if candidate.is_absolute() else base_dir / candidate

        patient_db = PatientDB.from_jsonl(resolve(cfg["paths"]["patient_db"]))
        graph = TCMKnowledgeGraph(load_triples(resolve(cfg["paths"]["triples"])))
        affinity = load_herb_target_affinity(resolve(cfg["paths"]["herb_targets"]))
        inference = cfg.get("inference", {})
        return cls(
            patient_db=patient_db,
            graph=graph,
            herb_target_affinity=affinity,
            max_herbs=int(inference.get("max_herbs", 7)),
            residual_threshold=float(inference.get("residual_threshold", 0.08)),
            pmi_threshold=float(inference.get("pmi_threshold", 0.0)),
            contraindication_penalty=float(inference.get("contraindication_penalty", 0.35)),
        )

    def infer(self, patient_id: str) -> InferenceResult:
        patient = self.patient_db.get(patient_id)
        symptoms = self.parser.normalize_symptoms(patient.symptoms)
        syndrome = self.diagnoser.predict(symptoms)
        targets = patient.targets or self.graph.targets_for_symptoms(
            list(symptoms),
            pmi_threshold=self.pmi_threshold,
        )

        ranked = self.policy.rank_candidates(
            patient=patient,
            symptoms=symptoms,
            syndrome=syndrome.syndrome,
            targets=targets,
        )

        residual = dict(symptoms)
        prescription: list[PrescriptionStep] = []
        selected: list[str] = []
        for step in range(1, self.max_herbs + 1):
            candidates = self.policy.rank_candidates(patient, residual, syndrome.syndrome, targets, selected)
            candidates = [candidate for candidate in candidates if not candidate.contraindications]
            if not candidates:
                break
            chosen = candidates[0]
            trace = self.policy.symptom_trace(chosen.herb, residual, targets)
            marginal_relief = min(sum(trace.values()), residual_norm(residual))
            residual = {
                symptom: max(0.0, severity - trace.get(symptom, 0.0))
                for symptom, severity in residual.items()
            }
            selected.append(chosen.herb)
            prescription.append(
                PrescriptionStep(
                    step=step,
                    herb=chosen.herb,
                    marginal_relief=round(marginal_relief, 6),
                    residual_symptoms={key: round(value, 6) for key, value in residual.items()},
                    trace=trace,
                    covered_targets=chosen.covered_targets,
                    provenance=chosen.provenance,
                )
            )
            if residual_norm(residual) <= self.residual_threshold:
                break

        herb_targets = [
            target
            for step in prescription
            for target in step.covered_targets
        ]
        notes = [
            "DMease provides research-oriented decision support; final diagnosis and prescription remain clinician-reviewed.",
            "The result is generated from the configured knowledge graph, compatibility constraints, and sequential herb policy.",
        ]
        return InferenceResult(
            patient_id=patient.patient_id,
            syndrome=syndrome,
            targets=targets,
            ranked_candidates=ranked,
            prescription=prescription,
            target_consistency_score=round(target_consistency_score(targets, herb_targets), 6),
            notes=notes,
        )

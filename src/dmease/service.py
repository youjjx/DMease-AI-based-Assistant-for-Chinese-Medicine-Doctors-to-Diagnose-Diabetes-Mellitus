from __future__ import annotations

from pathlib import Path

from .constraints import DEFAULT_SAFETY_NOTES
from .diagnosis import SyndromeReasoner
from .knowledge_graph import KnowledgeGraph
from .patient_db import PatientDatabase
from .recommender import HybridHerbRecommender
from .symptom_parser import SymptomParser
from .types import AnalysisResult, PatientProfile


class DMeaseService:
    """Facade used by Streamlit, tests, and future API integrations."""

    def __init__(
        self,
        graph_path: str | Path = "data/knowledge_graph.json",
        database_path: str | Path = "data/patients.db",
        checkpoint_path: str | Path = "checkpoints/kan_ppo.pt",
    ):
        self.graph = KnowledgeGraph(graph_path)
        self.parser = SymptomParser(self.graph.symptoms)
        self.reasoner = SyndromeReasoner(self.graph)
        self.recommender = HybridHerbRecommender(self.graph, checkpoint_path)
        self.database = PatientDatabase(database_path)

    def analyze(
        self,
        patient: PatientProfile,
        selected_symptoms: list[str] | None = None,
        top_herbs: int = 8,
    ) -> AnalysisResult:
        parsed = self.parser.parse(patient.complaint, selected_symptoms)
        present = [item.canonical for item in parsed if item.present]
        if present:
            syndromes = self.reasoner.infer(parsed, patient)
            recommendations, excluded = self.recommender.recommend(
                syndromes, present, patient, top_k=top_herbs
            )
        else:
            syndromes, recommendations, excluded = [], [], {}
        return AnalysisResult(
            patient=patient,
            parsed_symptoms=parsed,
            syndromes=syndromes,
            recommendations=recommendations,
            excluded_herbs=excluded,
            safety_notes=list(DEFAULT_SAFETY_NOTES),
        )

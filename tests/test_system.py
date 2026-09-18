from __future__ import annotations

from pathlib import Path

from dmease.service import DMeaseService
from dmease.types import PatientProfile


GRAPH = Path(__file__).parents[1] / "data" / "knowledge_graph.json"


def make_service(tmp_path: Path) -> DMeaseService:
    return DMeaseService(GRAPH, tmp_path / "patients.db", tmp_path / "missing.pt")


def test_symptom_normalization_and_negation(tmp_path):
    service = make_service(tmp_path)
    parsed = service.parser.parse("最近很口渴，也容易疲劳，但无手脚麻木。")
    states = {item.canonical: item.present for item in parsed}
    assert states["口渴多饮"] is True
    assert states["乏力"] is True
    assert states["肢体麻木"] is False


def test_end_to_end_reasoning_has_explanations(tmp_path):
    service = make_service(tmp_path)
    patient = PatientProfile(
        name="P001",
        constitution="阴虚质",
        complaint="口渴多饮，口干咽燥，多食易饥，形体消瘦。",
    )
    result = service.analyze(patient)
    assert result.syndromes[0].name == "阴虚燥热证"
    assert result.recommendations
    assert result.recommendations[0].evidence_paths


def test_safety_constraints_exclude_pregnancy_and_interactions(tmp_path):
    service = make_service(tmp_path)
    patient = PatientProfile(
        name="P002",
        sex="女",
        pregnant=True,
        medications=["华法林"],
        complaint="肢体麻木，固定刺痛，视物模糊。",
    )
    result = service.analyze(patient)
    assert "丹参" in result.excluded_herbs
    assert "川芎" in result.excluded_herbs
    assert all(item.name not in {"丹参", "川芎"} for item in result.recommendations)


def test_patient_database_round_trip(tmp_path):
    service = make_service(tmp_path)
    result = service.analyze(PatientProfile(name="P003", complaint="乏力气短，口干。"))
    record_id = service.database.save(result)
    stored = service.database.get(record_id)
    assert stored is not None
    assert stored["patient"]["name"] == "P003"
    assert service.database.list()[0]["id"] == record_id

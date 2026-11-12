from pathlib import Path

from dmease.service import LegacyDMeaseService as DMeaseService
from dmease.types import PatientProfile

ROOT = Path(__file__).parents[1]


def test_packaged_nonclinical_checkpoint_is_loaded_and_used(tmp_path):
    service = DMeaseService(ROOT / "data/knowledge_graph.json", tmp_path / "patients.db",
                           ROOT / "models/nonclinical_kan_ppo.pt")
    assert service.recommender.policy is not None
    result = service.analyze(PatientProfile(complaint="口渴多饮，容易疲劳", allergies=["黄芪"]))
    assert result.metadata["ranking_mode"] == "图谱与KAN混合排序"
    assert result.metadata["model"]["data_scope"] == "nonclinical-graph-derived"
    assert len(result.metadata["model"]["checkpoint_sha256"]) == 64
    assert all(x.name != "黄芪" for x in result.recommendations)
    assert service.recommender._policy_scores(result.syndromes)


def test_broken_checkpoint_keeps_graph_workflow_available(tmp_path):
    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"not a checkpoint")
    service = DMeaseService(ROOT / "data/knowledge_graph.json", tmp_path / "patients.db", bad)
    assert service.recommender.policy is None
    assert "无法加载" in service.recommender.load_message
    assert service.analyze(PatientProfile(complaint="口渴多饮")).recommendations

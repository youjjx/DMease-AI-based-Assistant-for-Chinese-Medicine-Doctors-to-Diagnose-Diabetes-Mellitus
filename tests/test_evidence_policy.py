import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dmease.evidence_policy import EvidenceEnvironment, EvidenceGraph, EvidencePolicy
from dmease.kan import KANPolicy
from dmease.service import DMeaseService
from dmease.types import PatientProfile


def test_public_service_never_falls_back_to_constructed_graph(tmp_path, evidence_path):
    with pytest.raises(FileNotFoundError):
        DMeaseService(tmp_path / "missing.json", tmp_path / "patients.db")
    invalid = json.loads(evidence_path.read_text(encoding="utf8"))
    invalid["symptom_herb"] = []
    evidence_path.write_text(json.dumps(invalid), encoding="utf8")
    with pytest.raises(ValueError, match="校验"):
        DMeaseService(evidence_path, tmp_path / "patients.db")


def test_sequential_state_reward_and_dynamic_pair_constraints(evidence_path):
    graph = EvidenceGraph(evidence_path)
    env = EvidenceEnvironment(graph)
    state = env.reset({"symptoms": {"口渴": 1, "乏力": 1}, "forbidden": ["黄芪"], "forbidden_pairs": [["麦冬", "山药"]]})
    assert not env.action_mask()[graph.hnames["黄芪"]]
    state2, reward, done = env.step(graph.hnames["麦冬"])
    assert reward > 0 and not torch.equal(state, state2)
    assert done and not env.action_mask()[graph.hnames["山药"]]
    assert env.steps[0]["new_targets"] == ["t1"]
    assert env.steps[0]["residual_evidence"]["乏力"] > 0
    with pytest.raises(ValueError):
        env.step(graph.hnames["黄芪"])


def test_abstention_negative_only_and_uncertain_evidence(tmp_path, evidence_path):
    service = DMeaseService(evidence_path, tmp_path / "patients.db", tmp_path / "missing.pt")
    assert not service.analyze(PatientProfile(complaint="口渴", pregnant=True)).recommendations
    assert not service.analyze(PatientProfile(complaint="无口渴、无乏力")).recommendations
    gap = service.analyze(PatientProfile(complaint="肢体麻木"))
    assert not gap.recommendations and gap.metadata["evidence_gaps"] == ["肢体麻木"]
    positive = service.analyze(PatientProfile(complaint="无乏力，口渴"))
    assert {x.canonical for x in positive.parsed_symptoms if x.present} == {"口渴"}


def test_no_constitution_effect_without_training_evidence(tmp_path, evidence_path):
    service = DMeaseService(evidence_path, tmp_path / "patients.db", tmp_path / "missing.pt")
    a = service.analyze(PatientProfile(complaint="口渴，乏力", constitution="阴虚质"))
    b = service.analyze(PatientProfile(complaint="口渴，乏力", constitution="气虚质"))
    assert [x.name for x in a.recommendations] == [x.name for x in b.recommendations]
    assert a.metadata["trace"]["steps"] == b.metadata["trace"]["steps"]


def test_policy_checkpoint_and_gradient_matches_finite_difference(tmp_path, evidence_path):
    torch.manual_seed(123)
    graph = EvidenceGraph(evidence_path)
    network = KANPolicy(graph.state_dim, graph.action_dim)
    path = tmp_path / "policy.pt"
    torch.save({"state_dict": network.state_dict(), "metadata": {"signature": graph.signature, "architecture": "KANPolicy"}}, path)
    engine = EvidencePolicy(graph, path)
    task = {"symptoms": {"口渴": 1, "乏力": 1}}
    result = engine.infer(task)
    assert engine.policy is not None and result["steps"]
    first = result["steps"][0]
    env = EvidenceEnvironment(graph); state = env.reset(task)
    index = len(graph.symptoms) + graph.snames["口渴"]
    mask = env.action_mask().unsqueeze(0)
    values = []
    for sign in (-1, 1):
        changed = state.clone(); changed[index] += sign * .001
        distribution, _ = engine.policy.distribution(changed.unsqueeze(0), mask)
        values.append(float(distribution.log_prob(torch.tensor([first["action"]])).detach()))
    assert first["gradient_attribution"]["口渴"] == pytest.approx((values[1] - values[0]) / .002, abs=.002)
    # A metadata mismatch keeps the real graph available, with explicit status.
    torch.save({"state_dict": network.state_dict(), "metadata": {"signature": "old"}}, path)
    incompatible = EvidencePolicy(graph, path)
    assert incompatible.policy is None and "不匹配" in incompatible.load_message
    assert incompatible.infer(task)["steps"]


def test_query_splits_are_disjoint_and_reproducible(evidence_path):
    path = Path(__file__).parents[1] / "scripts/train_evidence_policy.py"
    spec = importlib.util.spec_from_file_location("training", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    graph = EvidenceGraph(evidence_path)
    tasks = module.build_tasks(graph)
    assert tasks == module.build_tasks(graph)
    train, validation, test = [set(t["id"] for t in tasks[k]) for k in ("train", "validation", "test")]
    assert not train & validation and not train & test and not validation & test


def test_validation_gate_and_explicit_model_comparison(tmp_path, evidence_path):
    graph = EvidenceGraph(evidence_path)
    network = KANPolicy(graph.state_dim, graph.action_dim)
    path = tmp_path / "policy.pt"
    torch.save({"state_dict": network.state_dict(), "metadata": {
        "signature": graph.signature, "architecture": "KANPolicy", "preferred_strategy": "greedy"}}, path)
    engine = EvidencePolicy(graph, path)
    task = {"symptoms": {"口渴": 1, "乏力": 1}}
    default = engine.infer(task)
    comparison = engine.infer(task, mode="kan")
    assert default["mode"] == "公开证据增益序贯选择"
    assert not default["steps"][0]["gradient_attribution"]
    assert comparison["mode"] == "公开证据KAN/PPO序贯选择"
    assert comparison["steps"][0]["gradient_attribution"]


def test_timestamp_changes_do_not_invalidate_semantic_evidence(tmp_path, evidence_path):
    data = json.loads(evidence_path.read_text(encoding="utf8"))
    old = EvidenceGraph(evidence_path).signature
    data["files"] = [{"retrieved_at": "different download time"}]
    evidence_path.write_text(json.dumps(data), encoding="utf8")
    assert EvidenceGraph(evidence_path).signature == old

import itertools
import numpy as np
import pytest

from dmease.evidence_policy import EvidenceGraph
from dmease.experiment_metrics import (graph_view, evaluate_reference, reference_score, exact_optimum,
                                     paired_statistics, holm_adjust)


def test_ablation_scores_same_reference_and_does_not_mutate(evidence_path):
    graph = EvidenceGraph(evidence_path)
    task = {"id": "query", "symptoms": {"口渴": 1, "乏力": 1}}
    view = graph_view(graph, "without_targets")
    assert graph.symptom_target.any() and not view.symptom_target.any()
    report = evaluate_reference(graph, view, [task])
    actions = [graph.hnames[x] for x in report["rows"][0]["selected_herbs"]]
    assert report["mean"]["objective"] == reference_score(graph, task, actions)["objective"]
    assert report["rows"][0]["target_evidence_available"]


def test_exact_enumeration_matches_scalar_subsets(evidence_path):
    graph = EvidenceGraph(evidence_path)
    task = {"id": "query", "symptoms": {"口渴": 1, "乏力": 1}, "forbidden_pairs": [["麦冬", "山药"]]}
    values = []
    for n in range(3):
        for subset in itertools.combinations(range(graph.action_dim), n):
            score = reference_score(graph, task, list(subset))
            if not score["hard_constraint_violations"]:
                values.append(score["objective"])
    assert exact_optimum(graph, task, max_steps=2)["objective"] == pytest.approx(max(values), abs=1e-6)


def test_missing_evidence_only_removes_observed_edges(evidence_path):
    graph = EvidenceGraph(evidence_path)
    view = graph_view(graph, drop_fraction=.5, seed=12)
    for name in ("direct", "symptom_target", "affiliation"):
        assert np.all(getattr(view, name) <= getattr(graph, name))
        assert np.array_equal(getattr(view, name), getattr(graph_view(graph, drop_fraction=.5, seed=12), name))


def test_paired_statistics_aligns_queries_and_handles_all_ties():
    a = [{"query_id": str(i), "objective": float(i)} for i in range(4)]
    b = list(reversed(a))
    report = paired_statistics(a, b, repetitions=100)
    assert report["mean_difference"] == 0 and report["sign_flip_p"] == 1
    assert report["wins_ties_losses"] == [0, 4, 0]
    with pytest.raises(ValueError):
        paired_statistics(a, b[:-1])
    assert holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])

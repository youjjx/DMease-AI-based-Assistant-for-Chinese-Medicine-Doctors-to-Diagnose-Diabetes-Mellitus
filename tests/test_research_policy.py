import numpy as np
import pytest
import torch

from dmease.evidence_policy import EvidenceGraph, EvidenceEnvironment
from dmease.research_policy import CubicSplineLayer, SplinePolicy, generalized_advantages, StablePPO, refine_selection


def test_cubic_basis_partition_derivative():
    layer = CubicSplineLayer(2, 3).double()
    x = torch.tensor([[-.86, .13], [.48, .94]], dtype=torch.float64, requires_grad=True)
    basis = layer.basis(x)
    assert torch.all(basis >= 0)
    assert torch.allclose(basis.sum(-1), torch.ones_like(x))
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)
    endpoints = layer.basis(torch.tensor([[-1., 1.]], dtype=torch.float64))
    assert torch.allclose(endpoints.sum(-1), torch.ones(1, 2, dtype=torch.float64))


def test_gae_terminal_identity():
    rewards, values = [.3, -.1, .4], [.2, .1, .3]
    adv = generalized_advantages(rewards, values, gamma=1, lam=1)
    assert np.allclose(np.array(adv)+values, [.6, .3, .4])


def test_shared_search_feasible_and_monotonic(evidence_path):
    graph = EvidenceGraph(evidence_path)
    task = {"symptoms": {"口渴": 1, "乏力": 1}, "forbidden": ["麦冬"], "forbidden_pairs": [["黄芪", "山药"]]}
    initial = [graph.hnames["黄芪"]]
    improved, info = refine_selection(graph, task, initial)
    env = EvidenceEnvironment(graph);env.reset(task)
    def score(indices):
        selected = np.zeros(graph.action_dim);selected[indices] = 1
        return env.coverage(selected)[2]["objective"]
    assert score(improved) >= score(initial)-1e-7
    assert graph.hnames["麦冬"] not in improved
    assert not {graph.hnames["黄芪"],graph.hnames["山药"]}.issubset(improved)
    assert info["evaluated_subsets"] > 0


def test_stable_ppo_finite_update_and_masks(evidence_path):
    graph = EvidenceGraph(evidence_path)
    policy = SplinePolicy(graph.state_dim, graph.action_dim)
    task = {"symptoms": {"口渴": 1, "乏力": 1}, "forbidden": ["麦冬"]}
    env = EvidenceEnvironment(graph, [task])
    trainer = StablePPO(policy)
    batch = trainer.collect(env, episodes=3)
    assert all(int(a) != graph.hnames["麦冬"] for a in batch[2])
    metrics = trainer.update(batch)
    assert all(np.isfinite(x) for x in metrics.values())

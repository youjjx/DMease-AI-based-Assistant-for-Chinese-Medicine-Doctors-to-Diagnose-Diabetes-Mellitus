"""Common-reference evaluation for research ablations and robustness tests."""
from __future__ import annotations

import copy
import itertools

import numpy as np
import torch
from torch import nn

from .evidence_policy import EvidenceEnvironment


class MLPPolicy(nn.Module):
    """Parameter-budget comparator, with the same actor/critic interface."""
    def __init__(self, state_dim, action_dim, hidden=272):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state):
        return self.actor(state), self.critic(state).squeeze(-1)

    def distribution(self, state, action_mask=None):
        logits, value = self(state)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=logits), value


def graph_view(graph, variant="full", drop_fraction=0, seed=42):
    view = copy.copy(graph)
    view.direct = graph.direct.copy()
    view.symptom_target = graph.symptom_target.copy()
    view.affiliation = graph.affiliation.copy()
    view.ppi = list(graph.ppi)
    if variant == "without_ppi":
        view.ppi = []
    elif variant == "without_direct":
        view.direct.fill(0)
    elif variant == "without_targets":
        view.symptom_target.fill(0); view.affiliation.fill(0); view.ppi = []
    elif variant not in ("full", "mlp", "warm_start_only", "ppo_only"):
        raise ValueError("Unknown experimental graph variant")
    if not 0 <= drop_fraction <= 1:
        raise ValueError("drop_fraction must be in [0,1]")
    rng = np.random.default_rng(seed)
    for matrix in (view.direct, view.symptom_target, view.affiliation):
        positions = np.argwhere(matrix > 0)
        n = int(round(len(positions) * drop_fraction))
        if n:
            removed = positions[rng.choice(len(positions), n, replace=False)]
            matrix[removed[:, 0], removed[:, 1]] = 0
    return view


def rollout(view, task, policy=None, mode="greedy", rng=None, **settings):
    env = EvidenceEnvironment(view, **settings)
    state = env.reset(task)
    while not env.done:
        mask = env.action_mask()
        if mode == "greedy":
            action = int(np.argmax(env.gains()))
        elif mode == "random":
            action = int(rng.choice(np.flatnonzero(mask.numpy())))
        else:
            with torch.no_grad():
                distribution, _ = policy.distribution(state.unsqueeze(0), mask.unsqueeze(0))
            action = int(distribution.probs.argmax())
        state, _, _ = env.step(action)
    return [x["action"] for x in env.steps], env.stop_reason


def reference_score(graph, task, actions):
    """No stepping through an ablated environment when measuring performance."""
    reference = EvidenceEnvironment(graph)
    reference.reset(task)
    chosen = np.zeros(graph.action_dim, dtype=np.float32)
    chosen[actions] = 1
    metrics = reference.coverage(chosen)[2]
    names = {graph.herbs[i]["name"] for i in actions}
    violations = len(names & set(task.get("forbidden", [])))
    pair_violations = sum(a in names and b in names for a, b in task.get("forbidden_pairs", []))
    upper_bound = reference.coverage(reference.allowed.astype(np.float32))[2]
    return {**metrics, "herb_count": len(actions), "hard_constraint_violations": violations + pair_violations,
            "duplicate_count": len(actions) - len(set(actions)),
            "target_evidence_available": bool(reference.relevant.any()),
            "ppi_evidence_available": bool(len(reference.relevant_edges)),
            "symptom_coverage_upper_bound_ignoring_pair_conflicts": upper_bound["symptom_evidence_coverage"]}


def summarize(rows):
    keys = ("objective", "symptom_evidence_coverage", "target_evidence_coverage", "ppi_edge_coverage",
            "herb_count", "hard_constraint_violations", "duplicate_count", "symptom_coverage_upper_bound_ignoring_pair_conflicts")
    result = {"queries": len(rows), "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys}}
    for channel in ("target", "ppi"):
        defined = [r for r in rows if r[channel + "_evidence_available"]]
        key = "target_evidence_coverage" if channel == "target" else "ppi_edge_coverage"
        result[channel + "_available_queries"] = len(defined)
        result[channel + "_coverage_when_available"] = float(np.mean([r[key] for r in defined])) if defined else None
    result["violation_total"] = sum(r["hard_constraint_violations"] for r in rows)
    return result


def evaluate_reference(reference, view, tasks, policy=None, mode="greedy", seed=42, **settings):
    rows = []
    rng = np.random.default_rng(seed)
    for task in tasks:
        actions, stop = rollout(view, task, policy, mode, rng, **settings)
        rows.append({"query_id": task["id"], "selected_herbs": [reference.herbs[i]["name"] for i in actions],
                     "stop_reason": stop, **reference_score(reference, task, actions)})
    return {**summarize(rows), "rows": rows}


def paired_statistics(left, right, seed=20261112, repetitions=10000):
    a = {r["query_id"]: r["objective"] for r in left}
    b = {r["query_id"]: r["objective"] for r in right}
    if not a or a.keys() != b.keys() or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired statistics require identical, unique query IDs")
    delta = np.array([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(0, len(delta), size=(repetitions, len(delta)))].mean(axis=1)
    permuted = (rng.choice([-1, 1], size=(2 * repetitions, len(delta))) * delta).mean(axis=1)
    observed = float(delta.mean())
    p = float((1 + np.sum(np.abs(permuted) >= abs(observed) - 1e-12)) / (len(permuted) + 1))
    sd = float(delta.std(ddof=1)) if len(delta) > 1 else 0
    return {"queries": len(delta), "mean_difference": observed,
            "query_bootstrap_95ci": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
            "sign_flip_p": p, "paired_dz": observed / sd if sd else (0 if not observed else None),
            "wins_ties_losses": [int((delta > 1e-6).sum()), int((np.abs(delta) <= 1e-6).sum()), int((delta < -1e-6).sum())]}


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = [0.] * len(p_values)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., float(p_values[i]) * (len(order) - rank)))
        adjusted[int(i)] = running
    return adjusted


def exact_optimum(graph, task, max_steps=8):
    """Enumerate all useful subsets, including initially nonpositive-gain actions."""
    env = EvidenceEnvironment(graph, max_steps=max_steps)
    env.reset(task)
    useful = ((env.initial @ graph.direct) > 0) | ((graph.affiliation @ env.relevant) > 0)
    candidates = np.flatnonzero(useful & env.allowed)
    best, best_actions, examined = 0., [], 1
    n = len(candidates)
    edges = env.relevant_edges
    tw = .3 if env.relevant.any() else 0
    pw = .05 if len(edges) else 0
    for start in range(1, 1 << n, 4096):
        integers = np.arange(start, min(start + 4096, 1 << n), dtype=np.int64)
        bits = ((integers[:, None] >> np.arange(n)) & 1).astype(np.float32)
        bits = bits[bits.sum(axis=1) <= max_steps]
        if not len(bits):
            continue
        selected = np.zeros((len(bits), graph.action_dim), dtype=np.float32)
        selected[:, candidates] = bits
        for a, b in task.get("forbidden_pairs", []):
            if a in graph.hnames and b in graph.hnames:
                selected = selected[~((selected[:, graph.hnames[a]] > 0) & (selected[:, graph.hnames[b]] > 0))]
        if not len(selected):
            continue
        targets = (selected @ graph.affiliation > 0).astype(np.float32)
        direct = (selected @ graph.direct.T > 0).astype(np.float32)
        indirect = (targets @ graph.symptom_target.T) / np.maximum(graph.symptom_target.sum(axis=1), 1)
        coverage = np.maximum(direct, indirect) @ env.initial / max(float(env.initial.sum()), 1e-8)
        tc = targets @ env.relevant / max(float(env.relevant.sum()), 1)
        pc = (targets[:, edges[:, 0]] * targets[:, edges[:, 1]]).mean(axis=1) if len(edges) else 0
        scores = (.65 * coverage + tw * tc + pw * pc) / (.65 + tw + pw) - .02 * selected.sum(axis=1)
        i = int(np.argmax(scores)); examined += len(selected)
        if float(scores[i]) > best + 1e-7:
            best, best_actions = float(scores[i]), np.flatnonzero(selected[i]).tolist()
    return {"query_id": task["id"], "objective": reference_score(graph, task, best_actions)["objective"],
            "selected_herbs": [graph.herbs[i]["name"] for i in best_actions], "enumerated_feasible_subsets": examined}

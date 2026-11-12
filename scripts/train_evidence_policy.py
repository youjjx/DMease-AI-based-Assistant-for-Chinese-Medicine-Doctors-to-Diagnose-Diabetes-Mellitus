"""Train/evaluate on disjoint public-evidence retrieval queries, not patients."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch

from dmease.evidence_policy import EvidenceGraph, EvidenceEnvironment, EvidencePolicy, canonical_hash
from dmease.kan import KANPolicy
from dmease.ppo import PPOTrainer


def build_tasks(graph):
    names = [x["name"] for x in graph.symptoms]
    sets = [(name,) for name in names] + list(itertools.combinations(names, 2)) + list(itertools.combinations(names, 3))
    tasks = {"train": [], "validation": [], "test": []}
    for group in sets:
        identity = canonical_hash(sorted(group))
        # Split by complete query identity BEFORE creating constraint variants.
        bucket = int(identity[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        # Bounded and reproducible scope: retain every single/pair and 1/8 triples.
        if len(group) == 3 and int(identity[8:16], 16) % 8:
            continue
        task = {"id": identity, "symptoms": dict.fromkeys(group, 1.0), "forbidden": [],
                "scope": "database-retrieval-query-not-a-clinical-record"}
        tasks[split].append(task)
    return tasks


def evaluate(graph, policy, tasks, mode="kan", seed=42):
    rows = []
    rng = np.random.default_rng(seed)
    for task in tasks:
        env = EvidenceEnvironment(graph)
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
        rows.append({"query_id": task["id"], **env.coverage()[2], "herb_count": len(env.steps),
                     "hard_constraint_violations": sum(x["herb"] in task.get("forbidden", []) for x in env.steps),
                     "stop_reason": env.stop_reason})
    keys = ["objective", "symptom_evidence_coverage", "target_evidence_coverage", "ppi_edge_coverage", "herb_count", "hard_constraint_violations"]
    return {"queries": len(rows), "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys}, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="data/processed/evidence_bundle.json")
    parser.add_argument("--output", default="models/public_evidence_kan_ppo.pt")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="models/public_evidence_evaluation.json")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    graph = EvidenceGraph(args.bundle)
    tasks = build_tasks(graph)
    env = EvidenceEnvironment(graph)
    actionable = []
    states, actions, masks = [], [], []
    for task in tasks["train"]:
        state = env.reset(task)
        if not env.done:
            actionable.append(task)
        while not env.done:
            action = int(np.argmax(env.gains()))
            states.append(state); actions.append(action); masks.append(env.action_mask())
            state, _, _ = env.step(action)
    if not actionable:
        raise ValueError("No actionable public-evidence queries")
    policy = KANPolicy(graph.state_dim, graph.action_dim)
    trainer = PPOTrainer(policy)
    sx, ay, am = torch.stack(states), torch.tensor(actions), torch.stack(masks)
    # Explicit supervised warm start on the TRAIN split's evidence-greedy policy.
    for epoch in range(args.warmup):
        order = torch.randperm(len(sx))
        for indexes in order.split(128):
            logits, _ = policy(sx[indexes])
            loss = torch.nn.functional.cross_entropy(logits.masked_fill(~am[indexes], -1e9), ay[indexes])
            trainer.optimizer.zero_grad(); loss.backward(); trainer.optimizer.step()
    environment = EvidenceEnvironment(graph, actionable, seed=args.seed)
    # Save on validation objective, keeping test queries untouched until selection.
    best_value, best_state, best_iteration = -float("inf"), None, 0
    validation_history = []
    for iteration in range(args.iterations + 1):
        if iteration:
            trainer.update(trainer.collect(environment, episodes=12))
        if iteration % 10 == 0 or iteration == args.iterations:
            validation = evaluate(graph, policy, tasks["validation"])
            value = validation["mean"]["objective"]
            validation_history.append({"iteration": iteration, "objective": value})
            if iteration > 0 and value > best_value:
                best_value, best_iteration = value, iteration
                best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            print(json.dumps(validation_history[-1]), flush=True)
    policy.load_state_dict(best_state)
    baseline_validation = evaluate(graph, policy, tasks["validation"], "greedy")["mean"]["objective"]
    reports = {mode: evaluate(graph, policy, tasks["test"], mode, args.seed) for mode in ("kan", "greedy", "random")}
    stress = [{**t, "forbidden": [graph.herbs[i % len(graph.herbs)]["name"]]} for i, t in enumerate(tasks["test"])]
    reports["kan_exclusion_stress"] = evaluate(graph, policy, stress)
    metadata = {"architecture": "KANPolicy", "signature": graph.signature, "evidence_fingerprint": graph.fingerprint,
        "data_scope": "public-database-associations", "training_tasks": "Evidence retrieval queries, not patient records",
        "seed": args.seed, "iterations": args.iterations, "warmup_epochs": args.warmup,
        "selected_iteration": best_iteration, "selection_metric": "validation evidence objective",
        "objective": "0.65 symptom association coverage + 0.30 target coverage + 0.05 PPI coverage, normalized for missing evidence; 0.02 herb cost",
        "state_dim": graph.state_dim, "action_dim": graph.action_dim,
        "clinical_validation": False, "constitution_adaptation": False}
    metadata.update(preferred_strategy="kan" if best_value >= baseline_validation else "greedy",
                    validation_objective=best_value, greedy_validation_objective=baseline_validation,
                    deployment_rule="Use KAN by default only when its validation objective meets the evidence-greedy baseline")
    trainer.save(args.output, metadata)
    report = {"metadata": metadata, "split_counts": {k: len(v) for k, v in tasks.items()},
        "split_ids": {k: [t["id"] for t in v] for k, v in tasks.items()},
        "task_definition": "All source-scoped single/pair queries plus deterministic 1/8 triples. Split by complete query hash; entities/edges are shared across splits. This evaluates query generalization, not unseen-edge or clinical generalization.",
        "validation_history": validation_history, "evaluation": reports,
        "limitations": ["Shared graph supplies both evidence and objective; this is not independent medical validation",
                        "No clinical accuracy, therapeutic benefit or learned patient-constitution claim",
                        "Greedy warm start is an engineering addition; paper-specific affinity reward not reproduced"]}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v["mean"] for k, v in reports.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

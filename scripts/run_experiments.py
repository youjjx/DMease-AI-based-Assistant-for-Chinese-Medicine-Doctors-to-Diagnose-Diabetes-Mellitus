"""Frozen-protocol ablations, common-reference scoring and robustness experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from dmease.evidence_policy import EvidenceGraph, EvidenceEnvironment, canonical_hash
from dmease.experiment_metrics import (MLPPolicy, graph_view, evaluate_reference,
    paired_statistics, holm_adjust, exact_optimum)
from dmease.kan import KANPolicy
from dmease.ppo import PPOTrainer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evidence_training", ROOT / "scripts/train_evidence_policy.py")
training = importlib.util.module_from_spec(spec); spec.loader.exec_module(training)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def provenance(protocol, graph):
    files = ["scripts/run_experiments.py", "scripts/train_evidence_policy.py", "src/dmease/experiment_metrics.py",
             "src/dmease/evidence_policy.py", "src/dmease/kan.py", "src/dmease/ppo.py"]
    return {"protocol_sha256": canonical_hash(protocol), "evidence_fingerprint": graph.fingerprint,
            "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
            "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
            "threads": 1, "device": "cpu"}


def train_variant(graph, tasks, protocol, variant, seed, record):
    torch.manual_seed(seed); np.random.seed(seed)
    view = graph_view(graph, variant)
    policy = (MLPPolicy if variant == "mlp" else KANPolicy)(graph.state_dim, graph.action_dim)
    trainer = PPOTrainer(policy)
    environment = EvidenceEnvironment(view)
    actionable, states, actions, masks = [], [], [], []
    for task in tasks["train"]:
        state = environment.reset(task)
        if not environment.done:
            actionable.append(task)
        while not environment.done:
            action = int(np.argmax(environment.gains()))
            states.append(state); actions.append(action); masks.append(environment.action_mask())
            state, _, _ = environment.step(action)
    if not actionable:
        raise ValueError("No actionable training tasks for variant")
    sx, ay, am = torch.stack(states), torch.tensor(actions), torch.stack(masks)
    warmup = 0 if variant == "ppo_only" else protocol["warmup_epochs"]
    for _ in range(warmup):
        for indexes in torch.randperm(len(sx)).split(128):
            logits, _ = policy(sx[indexes])
            loss = torch.nn.functional.cross_entropy(logits.masked_fill(~am[indexes], -1e9), ay[indexes])
            trainer.optimizer.zero_grad(); loss.backward(); trainer.optimizer.step()
    steps = 0 if variant == "warm_start_only" else protocol["ppo_iterations"]
    environment = EvidenceEnvironment(view, actionable, seed=seed)
    best, state_dict, best_iteration = -float("inf"), None, None
    history = []
    for iteration in range(steps + 1):
        if iteration:
            trainer.update(trainer.collect(environment, episodes=protocol["episodes_per_iteration"]))
        if iteration % protocol["validation_interval"] == 0 or iteration == steps:
            validation = evaluate_reference(graph, view, tasks["validation"], policy, "model")
            value = validation["mean"]["objective"]
            history.append({"iteration": iteration, "reference_objective": value})
            if (iteration > 0 or not steps) and value > best:
                best, best_iteration = value, iteration
                state_dict = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(state_dict)
    test = evaluate_reference(graph, view, tasks["test"], policy, "model")
    checkpoint = ROOT / f"outputs/experiments/{record['protocol_sha256'][:12]}/{variant}_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "metadata": {"signature": canonical_hash([graph.signature, variant, record]),
               "architecture": type(policy).__name__, "scope": "experiment-only", "variant": variant, "seed": seed}}, checkpoint)
    return {"variant": variant, "seed": seed, "provenance": record,
            "architecture": type(policy).__name__, "parameters": sum(x.numel() for x in policy.parameters()),
            "actionable_training_queries": len(actionable), "teacher_states": len(states),
            "warmup_epochs": warmup, "ppo_iterations": steps, "selected_iteration": best_iteration,
            "validation_history": history, "test": test,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def average_seed_rows(results):
    ids = [r["query_id"] for r in results[0]["test"]["rows"]]
    output = []
    for q in ids:
        rows = [next(x for x in result["test"]["rows"] if x["query_id"] == q) for result in results]
        output.append({"query_id": q, "objective": float(np.mean([x["objective"] for x in rows]))})
    return output


def aggregate(graph, tasks, protocol, record):
    result_dir = ROOT / "experiments/results"
    variants = {}
    for name in protocol["variants"]:
        values = [json.loads((result_dir / f"{name}_{seed}.json").read_text(encoding="utf8")) for seed in protocol["seeds"]]
        if any(v["provenance"] != record for v in values):
            raise ValueError("Experiment inputs/code changed; rerun variant results before aggregation")
        variants[name] = values
    summary = {}
    for name, values in variants.items():
        keys = values[0]["test"]["mean"]
        summary[name] = {"parameters": values[0]["parameters"],
            "seed_mean": {k: float(np.mean([v["test"]["mean"][k] for v in values])) for k in keys},
            "seed_std": {k: float(np.std([v["test"]["mean"][k] for v in values], ddof=1)) for k in keys},
            "target_available_queries": values[0]["test"]["target_available_queries"],
            "ppi_available_queries": values[0]["test"]["ppi_available_queries"],
            "target_coverage_when_available": [v["test"]["target_coverage_when_available"] for v in values],
            "ppi_coverage_when_available": [v["test"]["ppi_coverage_when_available"] for v in values]}
    comparisons = {name: paired_statistics(average_seed_rows(variants["full"]), average_seed_rows(variants[name]))
                   for name in protocol["variants"] if name != "full"}
    for name, corrected in zip(comparisons, holm_adjust([v["sign_flip_p"] for v in comparisons.values()])):
        comparisons[name]["holm_p"] = corrected
    # Descriptive parameter sweeps never choose or update production settings.
    sensitivity = []
    for parameter, values in protocol["sensitivity"].items():
        for value in values:
            report = evaluate_reference(graph, graph, tasks["test"], **{parameter: value})
            sensitivity.append({"parameter": parameter, "value": value, **report})
    baseline = evaluate_reference(graph, graph, tasks["test"])
    missing = []
    constraints = []
    for seed in protocol["seeds"]:
        checkpoint = ROOT / f"outputs/experiments/{record['protocol_sha256'][:12]}/full_{seed}.pt"
        policy = KANPolicy(graph.state_dim, graph.action_dim)
        policy.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
        policy.eval()
        for fraction in protocol["missing_evidence"]["drop_fractions"]:
            view = graph_view(graph, drop_fraction=fraction, seed=seed)
            for mode in ("greedy", "model"):
                missing.append({"seed": seed, "drop_fraction": fraction, "mode": mode,
                                **evaluate_reference(graph, view, tasks["test"], policy, mode)})
        for scenario in protocol["constraints"]:
            modified = []
            for task, row in zip(tasks["test"], baseline["rows"]):
                chosen = row["selected_herbs"]
                new = {**task, "forbidden": [], "forbidden_pairs": []}
                if scenario == "exclude_first_baseline_choice":
                    new["forbidden"] = chosen[:1]
                elif scenario == "exclude_all":
                    new["forbidden"] = list(graph.hnames)
                elif scenario == "only_one_allowed":
                    keep = chosen[0] if chosen else graph.herbs[0]["name"]
                    new["forbidden"] = [name for name in graph.hnames if name != keep]
                elif len(chosen) >= 2:
                    new["forbidden_pairs"] = [chosen[:2]]
                modified.append(new)
            for mode in ("greedy", "model"):
                constraints.append({"seed": seed, "scenario": scenario, "mode": mode,
                    "active_constraint_queries": sum(bool(t["forbidden"] or t["forbidden_pairs"]) for t in modified),
                    **evaluate_reference(graph, graph, modified, policy, mode)})
    exact = []
    for i, task in enumerate(tasks["test"]):
        exact.append(exact_optimum(graph, task))
        if i % 10 == 0:
            print(f"Exact subset audit {i + 1}/{len(tasks['test'])}", flush=True)
    save_json(result_dir / "sensitivity.json", {"provenance": record, "results": sensitivity})
    save_json(result_dir / "missing_evidence.json", {"provenance": record, "results": missing})
    save_json(result_dir / "constraints.json", {"provenance": record, "results": constraints})
    save_json(result_dir / "optimality.json", {"provenance": record, "exact": exact, "greedy": baseline,
              "exact_minus_greedy": paired_statistics(exact, baseline["rows"]),
              "exact_minus_full_kan": paired_statistics(exact, average_seed_rows(variants["full"]))})
    report = {"provenance": record, "protocol": protocol, "split_counts": {k: len(v) for k, v in tasks.items()},
              "ablations": summary, "full_minus_variant": comparisons,
              "baseline_greedy": {k: v for k, v in baseline.items() if k != "rows"},
              "constraints_evaluated": sum(r["queries"] for r in constraints),
              "constraint_violations": sum(r["violation_total"] for r in constraints),
              "exact_subsets_evaluated": sum(x["enumerated_feasible_subsets"] for x in exact),
              "exact_minus_greedy": paired_statistics(exact, baseline["rows"]),
              "completed_at": datetime.now(timezone.utc).isoformat()}
    save_json(result_dir / "summary.json", report)
    print(json.dumps({k: v["seed_mean"]["objective"] for k, v in summary.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if not args.aggregate and args.seed is None:
        parser.error("Supply --seed or --aggregate")
    torch.set_num_threads(1)
    protocol = json.loads((ROOT / "experiments/protocol.json").read_text(encoding="utf8"))
    graph = EvidenceGraph(ROOT / "data/processed/evidence_bundle.json")
    if graph.fingerprint != protocol["evidence_fingerprint"]:
        raise ValueError("Evidence does not match the frozen experiment protocol")
    tasks = training.build_tasks(graph)
    record = provenance(protocol, graph)
    if args.aggregate:
        aggregate(graph, tasks, protocol, record)
        return
    if args.seed not in protocol["seeds"]:
        parser.error("Seed not in experiment protocol")
    for variant in protocol["variants"]:
        destination = ROOT / f"experiments/results/{variant}_{args.seed}.json"
        if destination.exists():
            old = json.loads(destination.read_text(encoding="utf8"))
            if old["provenance"] == record:
                print(f"Verified cached experiment {variant}/{args.seed}", flush=True)
                continue
        started = time.perf_counter()
        result = train_variant(graph, tasks, protocol, variant, args.seed, record)
        result["elapsed_seconds"] = time.perf_counter() - started
        save_json(destination, result)
        print(f"Completed {variant}/{args.seed}: {result['test']['mean']['objective']:.6f}, {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

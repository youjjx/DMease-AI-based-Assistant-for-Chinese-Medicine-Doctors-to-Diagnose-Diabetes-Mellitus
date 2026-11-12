"""Frozen v2 optimization study with unseen query combinations and paper metrics."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from dmease.evidence_policy import EvidenceGraph, EvidenceEnvironment, canonical_hash
from dmease.experiment_metrics import MLPPolicy, exact_optimum, rollout, reference_score, summarize, paired_statistics, holm_adjust
from dmease.kan import KANPolicy
from dmease.research_policy import SplinePolicy, StablePPO, refine_selection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v2"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def fingerprint(protocol, graph):
    files = ["scripts/research_v2.py", "src/dmease/research_policy.py", "src/dmease/evidence_policy.py", "src/dmease/experiment_metrics.py", "src/dmease/kan.py"]
    return {"protocol": canonical_hash(protocol), "evidence": graph.fingerprint,
            "source_lf_sha256": {p: hashlib.sha256((ROOT / p).read_text(encoding="utf8").encode()).hexdigest() for p in files},
            "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu", "threads": 1}


def prepare(graph, protocol):
    old = read(ROOT / "experiments/queries.json")["splits"]
    used = {q["id"] for rows in old.values() for q in rows}
    candidates = []
    for names in itertools.combinations([s["name"] for s in graph.symptoms], 3):
        identity = canonical_hash(sorted(names))
        if identity not in used:
            candidates.append({"id": identity, "symptoms": dict.fromkeys(names, 1.), "forbidden": [], "scope": "public-retrieval-not-clinical"})
    candidates.sort(key=lambda q: hashlib.sha256(("v2:" + q["id"]).encode()).hexdigest())
    tasks = {"train": old["train"], "validation": candidates[:96], "test": candidates[96:288]}
    assert not ({q["id"] for q in tasks["validation"] + tasks["test"]} & used)
    save(OUT / "queries.json", {"evidence": graph.fingerprint, "splits": tasks})
    teacher = []
    for i, task in enumerate(tasks["train"]):
        teacher.append(exact_optimum(graph, task))
        if i % 50 == 0:
            print(f"Training-only exact teacher {i+1}/{len(tasks['train'])}", flush=True)
    save(OUT / "training_teacher.json", {"evidence": graph.fingerprint, "rows": teacher})


def teaching_data(graph, tasks, oracle=None):
    states, masks, targets, returns = [], [], [], []
    fallbacks = 0
    for task in tasks:
        env = EvidenceEnvironment(graph); state = env.reset(task)
        trajectory = []
        optimal = {graph.hnames[h] for h in oracle[task["id"]]["selected_herbs"]} if oracle else set()
        while not env.done:
            gains, mask = env.gains(), env.action_mask()
            target = torch.zeros(graph.action_dim, dtype=torch.bool)
            choices = [i for i in optimal if mask[i]] if oracle else []
            if not choices:
                if oracle:
                    fallbacks += 1
                choices = [int(np.argmax(gains))]
            target[choices] = True
            action = max(choices, key=lambda i: (gains[i], -i))
            next_state, reward, _ = env.step(action)
            trajectory.append((state, mask, target, reward))
            state = next_state
        total = 0.
        for state, mask, target, reward in reversed(trajectory):
            total += reward
            states.append(state); masks.append(mask); targets.append(target); returns.append(total)
    return torch.stack(states), torch.stack(masks), torch.stack(targets), torch.tensor(returns), fallbacks


def policy_for(variant, graph):
    cls = KANPolicy if variant in ("stable_rbf", "legacy") else MLPPolicy if variant == "oracle_mlp" else SplinePolicy
    return cls(graph.state_dim, graph.action_dim)


def evaluate(graph, tasks, policy=None):
    rows = []
    for task in tasks:
        actions, stop = rollout(graph, task, policy, "model" if policy else "greedy")
        rows.append({"query_id": task["id"], "selected_herbs": [graph.herbs[i]["name"] for i in actions],
                     "stop_reason": stop, **reference_score(graph, task, actions)})
    return {**summarize(rows), "rows": rows}


def train(graph, tasks, protocol, record, seed, variant):
    torch.manual_seed(seed); np.random.seed(seed)
    config = protocol["training"]
    oracle = {r["query_id"]: r for r in read(OUT / "training_teacher.json")["rows"]} if variant.startswith("oracle") else None
    states, masks, targets, returns, fallbacks = teaching_data(graph, tasks["train"], oracle)
    policy = policy_for(variant, graph)
    actor = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    for _ in range(config["warmup_epochs"]):
        for idx in torch.randperm(len(states)).split(128):
            logits = policy.actor(states[idx]).masked_fill(~masks[idx], -1e9)
            loss = -(torch.logsumexp(logits.masked_fill(~targets[idx], -1e9), -1) - torch.logsumexp(logits, -1)).mean()
            actor.zero_grad(); loss.backward(); actor.step()
    with torch.no_grad():
        critic_error_before = float((policy.critic(states).squeeze(-1)-returns).square().mean())
    critic = torch.optim.Adam(policy.critic.parameters(), lr=config["critic_lr"])
    for _ in range(config["critic_warmup_epochs"]):
        for idx in torch.randperm(len(states)).split(128):
            loss = (policy.critic(states[idx]).squeeze(-1)-returns[idx]).square().mean()
            critic.zero_grad(); loss.backward(); critic.step()
    with torch.no_grad():
        critic_error_after = float((policy.critic(states).squeeze(-1)-returns).square().mean())
        hidden = policy.actor[1](policy.actor[0](states))
        outside = float((hidden.abs()>1).float().mean())
    trainer = StablePPO(policy, config["actor_lr"], config["critic_lr"], config["target_kl"])
    actionable = []
    for task in tasks["train"]:
        env = EvidenceEnvironment(graph); env.reset(task)
        if not env.done:
            actionable.append(task)
    env = EvidenceEnvironment(graph, actionable, seed=seed)
    best, best_state, selected = -float("inf"), None, None
    histories, diagnostics = [], []
    for iteration in range(config["ppo_iterations"] + 1):
        if iteration:
            diagnostic = trainer.update(trainer.collect(env, config["episodes"]),
                         config["initial_entropy"] * (1 - iteration/config["ppo_iterations"]))
            diagnostics.append({"iteration": iteration, **diagnostic})
        if iteration % config["validation_interval"] == 0:
            report = evaluate(graph, tasks["validation"], policy)
            objective = report["mean"]["objective"]
            histories.append({"iteration": iteration, "objective": objective, "herb_count": report["mean"]["herb_count"]})
            if objective > best + 1e-9:
                best, selected = objective, iteration
                best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    checkpoint = ROOT / f"outputs/research_v2/{record['protocol'][:12]}/{variant}_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": {"architecture": type(policy).__name__, "variant": variant,
        "signature": canonical_hash([graph.signature, record, variant]), "scope": "experiment-only"}}, checkpoint)
    return {"variant": variant, "seed": seed, "provenance": record, "parameters": sum(x.numel() for x in policy.parameters()),
            "selected_iteration": selected, "validation_objective": best, "validation_history": histories,
            "ppo_diagnostics": diagnostics, "teacher_states": len(states), "teacher_fallbacks": fallbacks,
            "critic_mse_before": critic_error_before, "critic_mse_after": critic_error_after,
            "normalized_hidden_outside_unit_grid": outside,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def target_metrics(graph, task, actions):
    symptom_ids = {graph.symptoms[graph.snames[n]]["id"] for n in task["symptoms"]}
    herb_ids = {graph.herbs[i]["id"] for i in actions}
    st = {e["target"] for e in graph.data["symptom_target"] if e["eligible"] and e["source"] in symptom_ids}
    ht = {e["target"] for e in graph.data["herb_target"] if e["eligible"] and e["source"] in herb_ids}
    allowed = set(graph.target_ids)
    def jaccard(a, b):
        return len(a & b)/len(a | b) if a | b else None
    return {"target_jaccard_shared49": jaccard(st & allowed, ht & allowed),
            "target_jaccard_all_eligible": jaccard(st, ht), "symptom_targets_all": len(st),
            "herb_targets_all": len(ht), "intersection_all": len(st & ht), "union_all": len(st | ht)}


def final_score(graph, tasks, report):
    raw, refined = [], []
    for task, row in zip(tasks, report["rows"]):
        assert task["id"] == row["query_id"]
        actions = [graph.hnames[h] for h in row["selected_herbs"]]
        raw.append({**row, **target_metrics(graph, task, actions)})
        improved, diagnostic = refine_selection(graph, task, actions)
        refined.append({"query_id": task["id"], "selected_herbs": [graph.herbs[i]["name"] for i in improved],
                        **reference_score(graph, task, improved), **target_metrics(graph, task, improved), **diagnostic})
    output = {}
    for name, rows in (("raw", raw), ("shared_local_search", refined)):
        output[name] = {**summarize(rows), "rows": rows}
        for key in ("target_jaccard_shared49", "target_jaccard_all_eligible"):
            defined = [r[key] for r in rows if r[key] is not None]
            output[name][key] = {"defined_queries": len(defined), "mean": float(np.mean(defined)) if defined else None}
    return output


def aggregate(graph, tasks, protocol, record):
    # Check that every selection is frozen before ANY new test evaluation occurs.
    selections = {(variant, seed): read(OUT / f"training/{variant}_{seed}.json")
                  for variant in protocol["variants"] for seed in protocol["seeds"]}
    if any(r["provenance"] != record for r in selections.values()):
        raise ValueError("Source/protocol changed after training")
    test = tasks["test"]
    reports = {"greedy": [final_score(graph, test, evaluate(graph, test))]}
    reports["legacy"] = []
    old_protocol = read(ROOT / "experiments/results/summary.json")["provenance"]["protocol_sha256"][:12]
    for variant in ["legacy", *protocol["variants"]]:
        reports.setdefault(variant, [])
        for seed in protocol["seeds"]:
            checkpoint = ROOT / (f"outputs/experiments/{old_protocol}/full_{seed}.pt" if variant == "legacy" else f"outputs/research_v2/{record['protocol'][:12]}/{variant}_{seed}.pt")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if variant != "legacy":
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selections[variant, seed]["checkpoint_sha256"]
            policy = policy_for(variant, graph); policy.load_state_dict(payload["state_dict"]); policy.eval()
            start = time.perf_counter()
            report = final_score(graph, test, evaluate(graph, test, policy))
            report["seed"] = seed; report["elapsed_including_shared_search_seconds"] = time.perf_counter()-start
            reports[variant].append(report)
            print(f"Test {variant}/{seed}: {report['raw']['mean']['objective']:.6f}", flush=True)
    exact = []
    for i, task in enumerate(test):
        exact.append(exact_optimum(graph, task))
        if i % 40 == 0:
            print(f"Final ceiling {i+1}/{len(test)}", flush=True)
    summary = {}
    mean_rows = {}
    for variant, runs in reports.items():
        summary[variant] = {}
        mean_rows[variant] = {}
        for mode in ("raw", "shared_local_search"):
            summary[variant][mode] = {"objective": float(np.mean([r[mode]["mean"]["objective"] for r in runs])),
                "objective_seed_std": float(np.std([r[mode]["mean"]["objective"] for r in runs], ddof=1)) if len(runs)>1 else 0.,
                "herb_count": float(np.mean([r[mode]["mean"]["herb_count"] for r in runs])),
                "target_jaccard_shared49": float(np.mean([r[mode]["target_jaccard_shared49"]["mean"] for r in runs])),
                "target_jaccard_all_eligible": float(np.mean([r[mode]["target_jaccard_all_eligible"]["mean"] for r in runs]))}
            mean_rows[variant][mode] = [{"query_id": q["id"], "objective": float(np.mean([r[mode]["rows"][i]["objective"] for r in runs]))} for i,q in enumerate(test)]
    comparisons = {variant: paired_statistics(mean_rows[variant]["raw"], mean_rows["greedy"]["raw"]) for variant in reports if variant != "greedy"}
    for variant, value in zip(comparisons, holm_adjust([v["sign_flip_p"] for v in comparisons.values()])):
        comparisons[variant]["holm_p"] = value
    audit = {"provenance": record, "split_counts": {k: len(v) for k,v in tasks.items()}, "results": summary,
             "raw_minus_greedy": comparisons, "exact_mean": float(np.mean([r["objective"] for r in exact])),
             "exact_minus_greedy": paired_statistics(exact, mean_rows["greedy"]["raw"]),
             "exact_minus_refined_greedy": paired_statistics(exact, mean_rows["greedy"]["shared_local_search"]),
             "selected_iterations": {v: [selections[v,s]["selected_iteration"] for s in protocol["seeds"]] for v in protocol["variants"]}}
    save(OUT / "results.json", {"provenance": record, "methods": reports, "exact": exact})
    save(OUT / "summary.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true"); parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(1)
    graph = EvidenceGraph(ROOT / "data/processed/evidence_bundle.json")
    protocol = read(OUT / "protocol.json")
    assert graph.fingerprint == protocol["evidence_fingerprint"]
    record = fingerprint(protocol, graph)
    if args.prepare:
        prepare(graph, protocol); return
    tasks = read(OUT / "queries.json")["splits"]
    if args.aggregate:
        aggregate(graph, tasks, protocol, record); return
    if args.seed not in protocol["seeds"]:
        parser.error("Choose --prepare, --aggregate or a protocol seed")
    for variant in protocol["variants"]:
        path = OUT / f"training/{variant}_{args.seed}.json"
        if path.exists() and read(path)["provenance"] == record:
            print(f"Cached {variant}/{args.seed}", flush=True); continue
        start = time.perf_counter()
        result = train(graph, tasks, protocol, record, args.seed, variant)
        result["elapsed_seconds"] = time.perf_counter()-start
        save(path, result)
        print(f"Selected {variant}/{args.seed}: val={result['validation_objective']:.6f}, iteration={result['selected_iteration']}, {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

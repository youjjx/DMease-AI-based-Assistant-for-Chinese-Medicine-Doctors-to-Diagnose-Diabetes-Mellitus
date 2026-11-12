"""Resume v2 only when both numerical records AND experimental weights exist."""
import argparse
import hashlib
import time

import torch

from research_v2 import ROOT, OUT, read, save, fingerprint, train, EvidenceGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--legacy", action="store_true", help="Rebuild v1 comparison weights without replacing published v1 results")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = read(OUT / "protocol.json")
    if args.seed not in protocol["seeds"]:
        parser.error("Seed is not part of the frozen protocol")
    graph = EvidenceGraph(ROOT / "data/processed/evidence_bundle.json")
    assert graph.fingerprint == protocol["evidence_fingerprint"]
    if args.legacy:
        import run_experiments as previous
        first = read(ROOT / "experiments/protocol.json")
        previous_record = previous.provenance(first, graph)
        tasks = previous.training.build_tasks(graph)
        for variant in ("full", "without_ppi"):
            result = previous.train_variant(graph, tasks, first, variant, args.seed, previous_record)
            save(ROOT / f"outputs/research_v2/reproduced_v1_{variant}_{args.seed}.json", result)
            print(f"Rebuilt v1 {variant}/{args.seed}", flush=True)
        return
    tasks = read(OUT / "queries.json")["splits"]
    record = fingerprint(protocol, graph)
    for variant in protocol["variants"]:
        report = OUT / f"training/{variant}_{args.seed}.json"
        checkpoint = ROOT / f"outputs/research_v2/{record['protocol'][:12]}/{variant}_{args.seed}.pt"
        if report.exists() and checkpoint.exists():
            old = read(report)
            if old["provenance"] == record and old["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                print(f"Verified {variant}/{args.seed}", flush=True)
                continue
        started = time.perf_counter()
        result = train(graph, tasks, protocol, record, args.seed, variant)
        result["elapsed_seconds"] = time.perf_counter() - started
        save(report, result)
        print(f"Reproduced {variant}/{args.seed}", flush=True)


if __name__ == "__main__":
    main()

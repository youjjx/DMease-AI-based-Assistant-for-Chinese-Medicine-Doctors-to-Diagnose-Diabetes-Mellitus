from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dmease.kan import KANPolicy, torch
from dmease.knowledge_graph import KnowledgeGraph
from dmease.ppo import HerbRecommendationEnv, PPOTrainer


def build_matrix(graph: KnowledgeGraph):
    matrix = torch.zeros(len(graph.syndrome_names), len(graph.herb_names))
    for herb_index, herb_name in enumerate(graph.herb_names):
        herb = graph.herb(herb_name)
        for syndrome, weight in herb.get("syndromes", {}).items():
            matrix[graph.syndrome_names.index(syndrome), herb_index] = float(weight)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DMease KAN policy with PPO on graph-derived tasks")
    parser.add_argument("--graph", default="data/knowledge_graph.json")
    parser.add_argument("--output", default="checkpoints/kan_ppo.pt")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required. Install the environment from environment.yml or requirements.txt.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    graph = KnowledgeGraph(args.graph)
    matrix = build_matrix(graph)
    environment = HerbRecommendationEnv(matrix)
    policy = KANPolicy(len(graph.syndrome_names) + len(graph.herb_names), len(graph.herb_names))
    trainer = PPOTrainer(policy)

    for iteration in range(1, args.iterations + 1):
        rollout = trainer.collect(environment, episodes=args.episodes)
        metrics = trainer.update(rollout)
        if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
            print(json.dumps({"iteration": iteration, **metrics}, ensure_ascii=False))

    trainer.save(
        Path(args.output),
        metadata={
            "graph_version": graph.metadata.get("version"),
            "syndromes": graph.syndrome_names,
            "herbs": graph.herb_names,
            "seed": args.seed,
        },
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

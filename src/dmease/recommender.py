from __future__ import annotations

from pathlib import Path
import hashlib
import pickle

from .constraints import SafetyConstraintEngine
from .kan import KANPolicy, torch
from .knowledge_graph import KnowledgeGraph
from .types import HerbRecommendation, PatientProfile, SyndromeScore


class HybridHerbRecommender:
    """Ranks herbs using graph evidence, optional KAN policy, and constraints."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        checkpoint: str | Path | None = None,
        constraint_engine: SafetyConstraintEngine | None = None,
    ):
        self.graph = graph
        self.constraints = constraint_engine or SafetyConstraintEngine()
        self.policy = None
        self.model_metadata = {}
        self.load_message = "未配置检查点，当前使用知识图谱评分。"
        checkpoint = Path(checkpoint) if checkpoint else None
        if torch is not None and checkpoint and checkpoint.exists():
            try:
                self._load_checkpoint(checkpoint)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError, EOFError):
                self.policy = None
                self.model_metadata = {}
                self.load_message = "检查点无法加载，当前使用知识图谱评分。请检查模型文件与版本。"

    def _load_checkpoint(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        compatible = (
            metadata.get("syndromes") == self.graph.syndrome_names
            and metadata.get("herbs") == self.graph.herb_names
            and metadata.get("graph_version") == self.graph.metadata.get("version")
        )
        if not compatible:
            self.load_message = "检查点实体或知识版本不匹配，当前使用知识图谱评分。"
            return
        policy = KANPolicy(len(self.graph.syndrome_names) + len(self.graph.herb_names), len(self.graph.herb_names))
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        self.policy = policy
        self.model_metadata = {**metadata, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        self.load_message = "已加载KAN/PPO检查点，图谱评分占85%，策略概率占15%。"

    @property
    def mode(self) -> str:
        return "图谱与KAN混合排序" if self.policy is not None else "知识图谱排序"

    def _policy_scores(self, syndromes: list[SyndromeScore]) -> dict[str, float]:
        if self.policy is None or torch is None:
            return {}
        syndrome_map = {item.name: item.confidence for item in syndromes}
        state = [syndrome_map.get(name, 0.0) for name in self.graph.syndrome_names]
        state.extend([0.0] * len(self.graph.herb_names))
        with torch.no_grad():
            logits, _ = self.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            probabilities = logits.softmax(-1).squeeze(0).tolist()
        return dict(zip(self.graph.herb_names, probabilities))

    def recommend(
        self,
        syndromes: list[SyndromeScore],
        symptoms: list[str],
        patient: PatientProfile,
        top_k: int = 8,
    ) -> tuple[list[HerbRecommendation], dict[str, list[str]]]:
        syndrome_confidence = {item.name: item.confidence for item in syndromes}
        policy_scores = self._policy_scores(syndromes)
        candidates: list[tuple[dict, float, list[str]]] = []
        excluded: dict[str, list[str]] = {}
        for herb in self.graph.herbs:
            reasons = self.constraints.evaluate(herb, patient)
            if reasons:
                excluded[herb["name"]] = reasons
                continue
            graph_score = sum(
                syndrome_confidence.get(name, 0.0) * float(weight)
                for name, weight in herb.get("syndromes", {}).items()
            )
            if graph_score <= 0:
                continue
            # Expert graph remains dominant until a validated checkpoint is supplied.
            score = 0.85 * graph_score + 0.15 * policy_scores.get(herb["name"], graph_score)
            matched = [name for name in herb.get("syndromes", {}) if name in syndrome_confidence]
            candidates.append((herb, score, matched))

        candidates.sort(key=lambda item: item[1], reverse=True)
        selected: list[tuple[dict, float, list[str]]] = []
        for candidate in candidates:
            conflicts = self.constraints.incompatibilities([candidate[0], *(item[0] for item in selected)])
            if candidate[0]["name"] in conflicts:
                excluded[candidate[0]["name"]] = conflicts[candidate[0]["name"]]
                continue
            selected.append(candidate)
            if len(selected) >= top_k:
                break

        recommendations = []
        for rank, (herb, score, matched) in enumerate(selected, start=1):
            paths = []
            for syndrome in matched[:2]:
                paths.extend(self.graph.paths(symptoms, syndrome, herb["name"]))
            paths.extend(self.graph.mechanism_paths(herb["name"])[:1])
            recommendations.append(
                HerbRecommendation(
                    name=herb["name"],
                    score=round(float(score), 4),
                    rank=rank,
                    actions=herb.get("actions", []),
                    matched_syndromes=matched,
                    evidence_paths=list(dict.fromkeys(paths))[:4],
                    warnings=herb.get("notes", []),
                )
            )
        return recommendations, excluded

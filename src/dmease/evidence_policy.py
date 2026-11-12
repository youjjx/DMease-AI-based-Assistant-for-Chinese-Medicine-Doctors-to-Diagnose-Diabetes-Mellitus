"""Shared training/inference environment over traceable public associations.

The objective is evidence coverage, never symptom relief or binding affinity.
Unknown associations remain unknown. No synthetic graph fallback is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .kan import KANPolicy


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class EvidenceGraph:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf8"))
        fingerprint = data.pop("fingerprint")
        if fingerprint != canonical_hash({k: v for k, v in data.items() if k != "files"}) or data["schema"] != "dmease-public-evidence-v1":
            raise ValueError("公开证据文件校验失败，请重新构建。")
        self.fingerprint = fingerprint
        self.data = data
        self.symptoms = sorted(data["symptoms"], key=lambda x: x["id"])
        self.herbs = sorted(data["herbs"], key=lambda x: x["id"])
        self.sidx = {x["id"]: i for i, x in enumerate(self.symptoms)}
        self.hidx = {x["id"]: i for i, x in enumerate(self.herbs)}
        self.snames = {x["name"]: i for i, x in enumerate(self.symptoms)}
        self.hnames = {x["name"]: i for i, x in enumerate(self.herbs)}
        # Keep all significant targets shared by observed symptom and herb evidence.
        symptom_targets = {x["target"] for x in data["symptom_target"] if x["eligible"]}
        herb_targets = {x["target"] for x in data["herb_target"] if x["eligible"]}
        self.target_ids = sorted(symptom_targets & herb_targets)
        self.tidx = {x: i for i, x in enumerate(self.target_ids)}
        s, h, t = len(self.symptoms), len(self.herbs), len(self.target_ids)
        self.direct = np.zeros((s, h), dtype=np.float32)
        self.symptom_target = np.zeros((s, t), dtype=np.float32)
        self.affiliation = np.zeros((h, t), dtype=np.float32)
        for edge in data["symptom_herb"]:
            self.direct[self.sidx[edge["source"]], self.hidx[edge["target"]]] = 1
        for edge in data["symptom_target"]:
            if edge["eligible"] and edge["target"] in self.tidx:
                self.symptom_target[self.sidx[edge["source"]], self.tidx[edge["target"]]] = 1
        for edge in data["herb_target"]:
            if edge["eligible"] and edge["target"] in self.tidx:
                self.affiliation[self.hidx[edge["source"]], self.tidx[edge["target"]]] = 1
        self.ppi = [(self.tidx[e["source"]], self.tidx[e["target"]]) for e in data["ppi"]
                    if e["source"] in self.tidx and e["target"] in self.tidx]
        # Original symptoms + residual evidence + target relevance/coverage + chosen/allowed.
        self.state_dim = 2 * s + 2 * t + 2 * h
        self.action_dim = h
        self.signature = canonical_hash({"bundle": fingerprint, "state_schema": "evidence-coverage-v1",
            "symptoms": list(self.sidx), "herbs": list(self.hidx), "targets": self.target_ids})

    def vector(self, symptoms):
        vector = np.zeros(len(self.symptoms), dtype=np.float32)
        for name, value in symptoms.items():
            if name in self.snames:
                if not np.isfinite(value) or value < 0:
                    raise ValueError("症状权重必须是非负有限数。")
                vector[self.snames[name]] = min(float(value), 2) / 2
        return vector

    def paths(self, symptom_names, herb_index):
        herb = self.herbs[herb_index]
        output = []
        for name in symptom_names:
            i = self.snames.get(name)
            if i is None:
                continue
            symptom = self.symptoms[i]
            if self.direct[i, herb_index]:
                output.append(f"{name} → SymMap关联 → {herb['name']}；{symptom['source_url']}")
            shared = np.flatnonzero(self.symptom_target[i] * self.affiliation[herb_index])
            for j in shared[:2]:
                target = self.data["targets"][self.target_ids[j]]["name"]
                output.append(f"{name} → {target} ← {herb['name']}；SymMap FDR(BH)<0.05关联；{herb['source_url']}")
        return output


class EvidenceEnvironment:
    def __init__(self, graph, tasks=None, seed=42, max_steps=8, herb_cost=.02, ppi_weight=.05):
        self.graph = graph
        self.tasks = tasks or []
        self.rng = np.random.default_rng(seed)
        self.max_steps, self.herb_cost, self.ppi_weight = max_steps, herb_cost, ppi_weight
        self.current_task = None

    def reset(self, task=None):
        if task is None:
            if not self.tasks:
                raise ValueError("No evidence retrieval tasks supplied")
            task = self.tasks[int(self.rng.integers(len(self.tasks)))]
        self.current_task = task
        self.initial = self.graph.vector(task["symptoms"])
        self.selected = np.zeros(self.graph.action_dim, dtype=np.float32)
        self.allowed = np.ones(self.graph.action_dim, dtype=bool)
        for name in task.get("forbidden", []):
            if name in self.graph.hnames:
                self.allowed[self.graph.hnames[name]] = False
        self.pairs = {tuple(x) for x in task.get("forbidden_pairs", [])}
        self.relevant = (self.initial @ self.graph.symptom_target > 0).astype(np.float32)
        edges = [(a, b) for a, b in self.graph.ppi if self.relevant[a] and self.relevant[b]]
        self.relevant_edges = np.asarray(edges, dtype=int).reshape(-1, 2)
        self.steps = []
        self.done = False
        self.stop_reason = ""
        self._stop_if_needed()
        return self.state()

    def coverage(self, selected=None):
        chosen = self.selected if selected is None else selected
        target_covered = (chosen @ self.graph.affiliation > 0).astype(np.float32)
        direct = (self.graph.direct @ chosen > 0).astype(np.float32)
        counts = self.graph.symptom_target.sum(axis=1)
        indirect = (self.graph.symptom_target @ target_covered) / np.maximum(counts, 1)
        symptoms = np.maximum(direct, indirect)
        symptom_score = float(self.initial @ symptoms / max(float(self.initial.sum()), 1e-8))
        target_score = float(target_covered @ self.relevant / max(float(self.relevant.sum()), 1))
        edges = self.relevant_edges
        ppi_score = float(np.mean(target_covered[edges[:, 0]] * target_covered[edges[:, 1]])) if len(edges) else 0
        # Renormalize when target/PPI evidence does not exist for this query.
        tw = .3 if self.relevant.any() else 0
        pw = self.ppi_weight if len(edges) else 0
        objective = (.65 * symptom_score + tw * target_score + pw * ppi_score) / (.65 + tw + pw)
        return symptoms, target_covered, {"symptom_evidence_coverage": symptom_score,
            "target_evidence_coverage": target_score, "ppi_edge_coverage": float(ppi_score),
            "objective": float(objective - self.herb_cost * chosen.sum())}

    def state(self):
        symptoms, targets, _ = self.coverage()
        return torch.tensor(np.concatenate([self.initial, self.initial * (1 - symptoms),
            self.relevant, targets * self.relevant, self.selected, self.hard_mask().astype(np.float32)]), dtype=torch.float32)

    def hard_mask(self):
        mask = self.allowed & (self.selected == 0)
        selected_names = {self.graph.herbs[i]["name"] for i in np.flatnonzero(self.selected)}
        for i, herb in enumerate(self.graph.herbs):
            if any((herb["name"], other) in self.pairs or (other, herb["name"]) in self.pairs for other in selected_names):
                mask[i] = False
        return mask

    def gains(self):
        baseline = self.coverage()[2]["objective"]
        values = np.full(self.graph.action_dim, -np.inf, dtype=np.float32)
        for index in np.flatnonzero(self.hard_mask()):
            selected = self.selected.copy()
            selected[index] = 1
            values[index] = self.coverage(selected)[2]["objective"] - baseline
        return values

    def action_mask(self):
        return torch.tensor(self.hard_mask() & (self.gains() > 1e-7))

    def _stop_if_needed(self):
        if self.selected.sum() >= self.max_steps:
            self.done, self.stop_reason = True, "达到候选数量上限"
        elif not self.initial.any():
            self.done, self.stop_reason = True, "没有已映射的阳性症状"
        elif not self.hard_mask().any():
            self.done, self.stop_reason = True, "没有符合约束的候选"
        elif not self.action_mask().any():
            self.done, self.stop_reason = True, "没有可增加证据覆盖的候选"

    def step(self, action):
        if self.done or not 0 <= action < self.graph.action_dim or not self.action_mask()[action]:
            raise ValueError("Invalid or masked evidence-selection action")
        before_s, before_t, before = self.coverage()
        state_before = self.state().tolist()
        hard_exclusions = [self.graph.herbs[i]["name"] for i in np.flatnonzero(~self.hard_mask() & (self.selected == 0))]
        self.selected[action] = 1
        after_s, after_t, after = self.coverage()
        reward = after["objective"] - before["objective"]
        self.steps.append({"step": len(self.steps) + 1, "herb": self.graph.herbs[action]["name"],
            "action": action, "marginal_evidence_gain": reward,
            "new_targets": [self.graph.target_ids[i] for i in np.flatnonzero((after_t - before_t) * self.relevant)],
            "residual_evidence": {x["name"]: float(self.initial[i] * (1 - after_s[i])) for i, x in enumerate(self.graph.symptoms) if self.initial[i]},
            "state_before": state_before, "metrics": after})
        self.steps[-1]["excluded_at_step"] = hard_exclusions
        self._stop_if_needed()
        return self.state(), reward, self.done


class EvidencePolicy:
    def __init__(self, graph, checkpoint=None):
        self.graph, self.policy = graph, None
        self.model_metadata = {}
        self.load_message = "使用公开证据增益逐步选择。"
        if checkpoint and Path(checkpoint).is_file():
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                metadata = payload["metadata"]
                if metadata.get("signature") != graph.signature or metadata.get("architecture") != "KANPolicy":
                    raise ValueError("模型与公开证据版本或状态结构不匹配")
                policy = KANPolicy(graph.state_dim, graph.action_dim)
                policy.load_state_dict(payload["state_dict"])
                if not all(torch.isfinite(parameter).all() for parameter in policy.parameters()):
                    raise ValueError("模型含非有限参数")
                policy.eval()
                self.policy = policy
                self.model_metadata = {**metadata, "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
                self.load_message = "已加载公开证据任务训练的KAN/PPO逐步策略。"
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError, EOFError) as error:
                self.load_message = f"检查点不可用：{error}；使用公开证据增益逐步选择。"

    @property
    def mode(self):
        return "公开证据KAN/PPO序贯选择" if self.policy is not None and self.model_metadata.get("preferred_strategy", "kan") == "kan" else "公开证据增益序贯选择"

    def infer(self, task, top_k=8, explain=True, mode=None):
        if mode not in (None, "greedy", "kan"):
            raise ValueError("未知策略")
        use_kan = self.policy is not None and (mode == "kan" or mode is None and self.model_metadata.get("preferred_strategy", "kan") == "kan")
        env = EvidenceEnvironment(self.graph, max_steps=top_k)
        state = env.reset(task)
        while not env.done:
            mask = env.action_mask()
            gradient = {}
            if use_kan:
                tensor = state.unsqueeze(0).detach().requires_grad_(explain)
                distribution, _ = self.policy.distribution(tensor, mask.unsqueeze(0))
                action = int(distribution.probs.argmax())
                probability = float(distribution.probs[0, action].detach())
                if explain:
                    grad = torch.autograd.grad(distribution.log_prob(torch.tensor([action])).sum(), tensor)[0][0]
                    gradient = {symptom["name"]: float(grad[len(self.graph.symptoms) + i].detach()) for i, symptom in enumerate(self.graph.symptoms) if env.initial[i]}
            else:
                action = int(np.argmax(env.gains()))
                probability = None
            state, _, _ = env.step(action)
            env.steps[-1].update(policy_probability=probability, gradient_attribution=gradient)
        _, _, metrics = env.coverage()
        return {"steps": env.steps, "stop_reason": env.stop_reason, "metrics": metrics,
                "signature": self.graph.signature, "data_scope": self.graph.data["data_scope"],
                "mode": "公开证据KAN/PPO序贯选择" if use_kan else "公开证据增益序贯选择",
                "strategy_notice": "未加载匹配模型，本次使用公开证据增益策略" if mode == "kan" and self.policy is None else "",
                "uncovered_symptoms": [x["name"] for i, x in enumerate(self.graph.symptoms)
                                       if env.initial[i] and env.coverage()[0][i] < 1 - 1e-6]}

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from .evidence_policy import EvidenceGraph, EvidencePolicy
from .patient_db import PatientDatabase
from .symptom_parser import SymptomParser
from .types import AnalysisResult, HerbRecommendation, PatientProfile, SyndromeScore


class PublicGraphView:
    """UI projection of actual association rows, with no handcrafted scores."""
    def __init__(self, evidence):
        self.evidence = evidence
        self.metadata = {"version": "SymMap2.0-" + evidence.fingerprint[:12]}
        self.symptoms = evidence.symptoms
        self.herbs = [{**x, "actions": [], "notes": []} for x in evidence.herbs]
        self._herb = {x["name"]: x for x in self.herbs}
        self.symptom_names = [x["name"] for x in evidence.symptoms]
        self.herb_names = [x["name"] for x in evidence.herbs]
        self.syndromes = []
        for sid, syndrome in evidence.data["syndromes"].items():
            linked = {evidence.symptoms[evidence.sidx[e["source"]]]["name"]: 1.0
                      for e in evidence.data["symptom_syndrome"] if e["target"] == sid}
            self.syndromes.append({**syndrome, "symptoms": linked,
                "description": f"SymMap 2.0证候关联；http://www.symmap.org/detail/{sid}"})
        self.syndrome_names = sorted(x["name"] for x in self.syndromes)

    def syndrome(self, name):
        return next(x for x in self.syndromes if x["name"] == name)

    def herb(self, name):
        return self._herb[name]

    def herbs_for_syndrome(self, name):
        # A two-hop shared-symptom path, explicitly not a syndrome-treats-herb claim.
        symptoms = self.syndrome(name)["symptoms"]
        counts = defaultdict(int)
        for symptom in symptoms:
            i = self.evidence.snames[symptom]
            for j, value in enumerate(self.evidence.direct[i]):
                if value:
                    counts[j] += 1
        return [{"herb": self.herbs[j], "weight": count} for j, count in sorted(counts.items(), key=lambda x: -x[1])]

    def mechanism_paths(self, name):
        j = self.evidence.hnames[name]
        return [f"{name} → SymMap靶点关联 → {self.evidence.data['targets'][target]['name']}；{self.herbs[j]['source_url']}"
                for target, i in self.evidence.tidx.items() if self.evidence.affiliation[j, i]][:20]

    def summary(self):
        return {"version": self.metadata["version"], "symptoms": len(self.symptoms), "syndromes": len(self.syndromes),
                "herbs": len(self.herbs), "targets": len(self.evidence.target_ids),
                "relations": sum(len(self.evidence.data[x]) for x in ("symptom_herb", "symptom_syndrome", "symptom_target", "herb_target"))}


class PublicEvidenceService:
    def __init__(self, graph_path="data/processed/evidence_bundle.json", database_path="runtime/patients.db",
                 checkpoint_path="models/public_evidence_kan_ppo.pt"):
        self.evidence = EvidenceGraph(graph_path)
        self.graph = PublicGraphView(self.evidence)
        self.parser = SymptomParser(self.graph.symptoms)
        self.recommender = EvidencePolicy(self.evidence, checkpoint_path)
        self.database = PatientDatabase(database_path)

    def diagnose(self, parsed):
        positive = {x.canonical: x.severity for x in parsed if x.present}
        denied = {x.canonical for x in parsed if not x.present}
        scores = []
        for syndrome in self.graph.syndromes:
            linked = set(syndrome["symptoms"])
            matched = sorted(linked & positive.keys())
            if not matched:
                continue
            # Relative evidence overlap, not a learned diagnosis probability.
            score = sum(positive[x] for x in matched) / max(sum(positive.values()), 1)
            score *= len(matched) / (len(matched) + len(linked & denied))
            scores.append((syndrome, score, matched))
        total = sum(x[1] for x in scores) or 1
        scores.sort(key=lambda x: (-x[1], x[0]["id"]))
        return [SyndromeScore(x["name"], score, score / total, matched, [], x["description"])
                for x, score, matched in scores[:3]]

    def analyze(self, patient: PatientProfile, selected_symptoms=None, top_herbs=8, strategy=None):
        if not patient.name.strip() or not 1 <= patient.age <= 120 or not 1 <= top_herbs <= 20:
            raise ValueError("请核对患者标识、年龄和候选数量。")
        allowed_selections = set(self.graph.symptom_names)
        if set(selected_symptoms or []) - allowed_selections:
            raise ValueError("所选症状不在当前公开证据版本中，请重新选择。")
        parsed = self.parser.parse(patient.complaint, selected_symptoms)
        observed = {x.canonical: x.severity for x in parsed if x.present}
        excluded = {}
        for herb in self.graph.herb_names:
            reasons = []
            if herb in patient.forbidden_herbs:
                reasons.append("本次明确禁用")
            if any(name.strip() and name.strip() in herb for name in patient.allergies):
                reasons.append("命中填写的过敏名称")
            if patient.pregnant:
                reasons.append("当前证据未提供妊娠适用性，本次停止候选生成")
            if reasons:
                excluded[herb] = reasons
        for pair in patient.forbidden_pairs:
            if len(pair) != 2 or pair[0] == pair[1] or any(x not in self.graph.herb_names for x in pair):
                raise ValueError("配伍排除项须为当前范围内的两味不同药材。")
        task = {"symptoms": observed, "forbidden": list(excluded), "forbidden_pairs": patient.forbidden_pairs}
        trace = self.recommender.infer(task, top_herbs, mode=strategy)
        chosen = {x["herb"] for x in trace["steps"]}
        for a, b in patient.forbidden_pairs:
            if a in chosen and b not in chosen:
                excluded.setdefault(b, []).append(f"与已选{a}命中本次配伍排除")
            if b in chosen and a not in chosen:
                excluded.setdefault(a, []).append(f"与已选{b}命中本次配伍排除")
        syndromes = self.diagnose(parsed)
        recommendations = []
        for step in trace["steps"]:
            recommendations.append(HerbRecommendation(step["herb"], step["marginal_evidence_gain"], step["step"], [], [],
                self.evidence.paths(observed, step["action"]), ["公开数据库关联候选；不提供剂量或疗效判断"]))
        gaps = [x for x in observed if not self.evidence.direct[self.evidence.snames[x]].any()
                and not self.evidence.symptom_target[self.evidence.snames[x]].any()]
        notes = ["结果是公开数据库证据检索和排序，不构成处方或独立诊断。",
                 "数据库推断关系、证据覆盖及策略概率均不代表治疗效果。",
                 "体质、年龄、合并情况和当前用药已保存，但尚无经验证的个体化适用性及完整药物相互作用模型；须由专业人员复核。",
                 "当前自动处理填写的过敏名称、明确禁用、配伍排除及妊娠停止规则；未命中不等于安全。"]
        return AnalysisResult(patient, parsed, syndromes, recommendations, excluded, notes, metadata={
            "analysis_id": uuid4().hex, "graph_version": self.graph.metadata["version"],
            "ranking_mode": trace["mode"], "top_herbs": top_herbs, "selected_symptoms": list(selected_symptoms or []),
            "requested_strategy": strategy,
            "data_scope": self.evidence.data["data_scope"], "evidence_fingerprint": self.evidence.fingerprint,
            "model": self.recommender.model_metadata, "trace": trace, "evidence_gaps": gaps,
            "personalization_scope": "Explicit exclusion masks; constitution and medical history not learned from public associations",
            "source_urls": sorted({x["source_url"] for x in self.evidence.symptoms if x["name"] in observed})})

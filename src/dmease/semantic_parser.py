from __future__ import annotations

import json
import re

from dmease.schemas import Triple


class SemanticParser:
    """Offline surface-to-concept parser with an LLM-compatible interface."""

    def __init__(self, synonym_map: dict[str, str] | None = None):
        self.synonym_map = synonym_map or {
            "多饮": "口渴多饮",
            "口干": "口渴多饮",
            "乏力": "倦怠乏力",
            "疲乏": "倦怠乏力",
            "小便频": "尿频",
            "尿多": "尿频",
        }

    def normalize_symptoms(self, symptoms: dict[str, float] | list[str]) -> dict[str, float]:
        if isinstance(symptoms, dict):
            items = symptoms.items()
        else:
            items = ((symptom, 1.0) for symptom in symptoms)

        normalized: dict[str, float] = {}
        for symptom, severity in items:
            canonical = self.normalize_term(str(symptom))
            normalized[canonical] = max(normalized.get(canonical, 0.0), float(severity))
        return normalized

    def normalize_term(self, term: str) -> str:
        text = re.sub(r"\s+", "", term)
        return self.synonym_map.get(text, text)

    def extract_triples_from_llm_json(self, raw_text: str) -> list[Triple]:
        """Parse JSON triplets emitted by an LLM prompt.

        The paper uses a Chinese-centric LLM to enforce JSON-formatted triples.
        This method accepts either a single JSON object or a JSON array, so the
        extraction backend can be swapped without changing the downstream graph.
        """

        payload = json.loads(raw_text)
        rows = payload if isinstance(payload, list) else [payload]
        return [Triple.from_dict(row) for row in rows]


def build_triplet_extraction_prompt(text: str) -> str:
    return f"""
你是中医糖尿病知识图谱抽取助手。请从输入文本中抽取三元组，输出 JSON 数组。
可用关系包括：
R1 Herb-affect by-Target
R2 Symptom-associated with-Target
R3 Symptom-indicates-Syndrome
R4 Syndrome-treatedBy-Herb
R5 Herb-contains-Compound
R6 Compound-targets-Gene
R7 Gene-associatedWith-Symptom
R8 Herb-contraindicatedWith-Herb
R9 Syndrome-hasStage-DiseaseStage
R10 Gene-participatesIn-Pathway

字段必须为 subject、predicate、object、source、confidence。

输入文本：
{text}
""".strip()


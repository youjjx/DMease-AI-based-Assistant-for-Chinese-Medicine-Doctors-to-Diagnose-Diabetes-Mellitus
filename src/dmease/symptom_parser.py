from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .types import ParsedSymptom


NEGATIONS = ("无", "没有", "未见", "否认", "不伴", "未出现")
SEVERITY = {
    "轻微": 0.6,
    "轻度": 0.7,
    "明显": 1.2,
    "较重": 1.3,
    "严重": 1.5,
    "剧烈": 1.6,
}


@dataclass(slots=True)
class LexiconItem:
    canonical: str
    aliases: list[str]


class SymptomParser:
    """Dictionary-based Chinese symptom normalizer with negation handling.

    The interface is deliberately model-agnostic: a future LLM extractor can
    return the same ``ParsedSymptom`` structure without changing downstream
    diagnosis or recommendation code.
    """

    def __init__(self, symptoms: list[dict]):
        self.lexicon = [
            LexiconItem(item["name"], sorted(set([item["name"], *item.get("aliases", [])]), key=len, reverse=True))
            for item in symptoms
        ]

    @staticmethod
    def _negated(text: str, start: int, end: int | None = None) -> bool:
        prefix = re.split(r"[，,。；;！？!?]|但是|但|然而|不过|却", text[:start])[-1]
        # These phrases negate a cause/quantifier, or negate the negation itself.
        # They are not an absence assertion about the following symptom.
        prefix = re.sub(r"(?:并非|并不是|不是)没有|不能否认|未否认|并非无|不无", "", prefix)
        prefix = re.sub(r"无(?:明显|特殊)?诱因|无论(?:何时|何地)?|无力", "", prefix)
        if end is not None:
            suffix = text[end:]
            if re.match(r"(?:症状)?(?:现已|已经|已)?(?:消失|不存在)", suffix):
                return True
        return any(token in prefix for token in NEGATIONS)

    @staticmethod
    def _severity(text: str, start: int, end: int) -> float:
        window = text[max(0, start - 5) : min(len(text), end + 5)]
        for token, value in SEVERITY.items():
            if token in window:
                return value
        return 1.0

    def parse(self, text: str, selected: list[str] | None = None) -> list[ParsedSymptom]:
        text = re.sub(r"\s+", "", (text or "").replace("\n", "，"))
        matches: dict[str, list[tuple[ParsedSymptom, int, bool]]] = defaultdict(list)
        spans = []
        for item in self.lexicon:
            for alias in item.aliases:
                for hit in re.finditer(re.escape(alias), text):
                    spans.append((hit.start(), hit.end(), alias, item.canonical))
        # Prefer a complete specific term (e.g. 夜尿频多) over its contained 尿频.
        accepted = []
        for start, end, alias, canonical in sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0], x[2] != x[3], x[3])):
            if any(a <= start and end <= b for a, b in accepted):
                continue
            accepted.append((start, end))
            prefix = re.split(r"[，,。；;！？!?]", text[:start])[-1]
            is_current = bool(re.search(r"目前|现在|今日|今天|现(?:已|无|未|有)", prefix))
            symptom = ParsedSymptom(canonical, alias, not self._negated(text, start, end),
                                    self._severity(text, start, end), text[max(0, start - 8):min(len(text), end + 8)])
            matches[canonical].append((symptom, start, is_current))

        for name in selected or []:
            matches[name].append((ParsedSymptom(name, name, True, 1.0, "医生勾选"), len(text) + 1, True))

        result: list[ParsedSymptom] = []
        for name, candidates in matches.items():
            current = [item for item in candidates if item[2]]
            if current:
                chosen = max(current, key=lambda item: item[1])[0]
            else:
                positives = [item for item in candidates if item[0].present]
                chosen = max(positives or candidates, key=lambda item: item[0].severity)[0]
            if len({item[0].present for item in candidates}) > 1:
                chosen.evidence += "；前后存在相反描述，请核对当前状态"
            result.append(chosen)
        return sorted(result, key=lambda item: (not item.present, item.canonical))

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
    def _negated(text: str, start: int) -> bool:
        prefix = text[max(0, start - 6) : start]
        return any(token in prefix for token in NEGATIONS)

    @staticmethod
    def _severity(text: str, start: int, end: int) -> float:
        window = text[max(0, start - 5) : min(len(text), end + 5)]
        for token, value in SEVERITY.items():
            if token in window:
                return value
        return 1.0

    def parse(self, text: str, selected: list[str] | None = None) -> list[ParsedSymptom]:
        text = re.sub(r"\s+", "", text or "")
        matches: dict[str, list[ParsedSymptom]] = defaultdict(list)

        for item in self.lexicon:
            for alias in item.aliases:
                for hit in re.finditer(re.escape(alias), text):
                    present = not self._negated(text, hit.start())
                    matches[item.canonical].append(
                        ParsedSymptom(
                            canonical=item.canonical,
                            original=alias,
                            present=present,
                            severity=self._severity(text, hit.start(), hit.end()),
                            evidence=text[max(0, hit.start() - 8) : min(len(text), hit.end() + 8)],
                        )
                    )

        for name in selected or []:
            if not any(item.present for item in matches.get(name, [])):
                matches[name].append(ParsedSymptom(name, name, True, 1.0, "医生勾选"))

        result: list[ParsedSymptom] = []
        for name, candidates in matches.items():
            # Explicit positive mention wins unless every mention is negated.
            positives = [item for item in candidates if item.present]
            chosen = max(positives or candidates, key=lambda item: item.severity)
            result.append(chosen)
        return sorted(result, key=lambda item: (not item.present, item.canonical))

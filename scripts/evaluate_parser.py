"""Development regression benchmark; these are linguistic fixtures, not EMRs."""
from pathlib import Path
import argparse
import hashlib
import json

from dmease.evidence_policy import EvidenceGraph
from dmease.symptom_parser import SymptomParser

ROOT = Path(__file__).resolve().parents[1]


def cases(graph):
    manual = [
        ("无明显诱因出现口渴", {"口渴": True}),
        ("乏力已消失，目前口渴", {"乏力": False, "口渴": True}),
        ("否认乏力，但口渴明显", {"乏力": False, "口渴": True}),
        ("夜尿频多", {"夜尿频多": True}),
        ("并非没有乏力", {"乏力": True}),
        ("没有乏力，口渴未缓解", {"乏力": False, "口渴": True}),
        ("无口渴及乏力", {"口渴": False, "乏力": False}),
        ("无乏力；口渴", {"乏力": False, "口渴": True}),
        ("既往口渴，现无口渴", {"口渴": False}),
        ("无论何时都感到乏力", {"乏力": True}),
        ("口渴已好转，但仍存在", {"口渴": True}),
        ("乏力不明显", {"乏力": True}),
        ("否认口渴\n乏力", {"口渴": False, "乏力": True}),
        ("口渴不存在，乏力持续", {"口渴": False, "乏力": True}),
        ("不伴肢体麻木，口渴", {"肢体麻木": False, "口渴": True}),
        ("不能否认存在乏力", {"乏力": True}),
        ("主诉：暂无可描述的不适", {}),
        ("从未出现乏力", {"乏力": False}),
        ("昨夜乏力，今天未见乏力", {"乏力": False}),
        ("口渴但没有乏力", {"口渴": True, "乏力": False}),
    ]
    output = [{"id": f"manual-{i:02}", "group": "curated_development", "text": text, "expected": expected}
              for i, (text, expected) in enumerate(manual)]
    templates = [("出现{term}", True), ("否认{term}", False), ("未见{term}", False),
                 ("{term}已消失", False), ("无明显诱因出现{term}", True), ("并非没有{term}", True)]
    for symptom in graph.symptoms:
        for i, (template, present) in enumerate(templates):
            output.append({"id": f"{symptom['id']}-{i}", "group": "generated_grammar",
                           "text": template.format(term=symptom["name"]), "expected": {symptom["name"]: present}})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("before", "after"))
    args = parser.parse_args()
    graph = EvidenceGraph(ROOT / "data/processed/evidence_bundle.json")
    rows = cases(graph)
    case_file = ROOT / "experiments/parser_cases.json"
    payload = {"scope": "Agent-authored development fixtures and generated grammar templates; no clinical records or expert labels", "cases": rows}
    if case_file.exists() and json.loads(case_file.read_text(encoding="utf8")) != payload:
        raise ValueError("Benchmark cases changed; do not compare to previously frozen results")
    if not case_file.exists():
        case_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    engine = SymptomParser(graph.symptoms)
    results = []
    for row in rows:
        predicted = {x.canonical: x.present for x in engine.parse(row["text"])}
        actual_pairs, expected_pairs = set(predicted.items()), set(row["expected"].items())
        results.append({**row, "predicted": predicted, "exact": predicted == row["expected"],
            "tp": len(actual_pairs & expected_pairs), "fp": len(actual_pairs - expected_pairs), "fn": len(expected_pairs - actual_pairs)})
    summaries = {}
    for group in ("all", "curated_development", "generated_grammar"):
        subset = [r for r in results if group == "all" or r["group"] == group]
        tp, fp, fn = [sum(r[k] for r in subset) for k in ("tp", "fp", "fn")]
        summaries[group] = {"cases": len(subset), "exact_matches": sum(r["exact"] for r in subset),
                            "entity_state_micro_f1": 2 * tp / max(2 * tp + fp + fn, 1), "fp": fp, "fn": fn}
    report = {"label": args.label, "case_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
              "parser_sha256": hashlib.sha256((ROOT / "src/dmease/symptom_parser.py").read_bytes()).hexdigest(),
              "evidence_fingerprint": graph.fingerprint, "scope": payload["scope"], "summary": summaries, "rows": results}
    out = ROOT / "experiments/results" / f"parser_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and args.label == "before":
        old = json.loads(out.read_text(encoding="utf8"))
        if old["parser_sha256"] != report["parser_sha256"]:
            raise ValueError("The frozen BEFORE result belongs to the original parser; do not overwrite it with a revised implementation")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()

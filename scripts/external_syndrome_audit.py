"""External frozen zero-shot audit; publish aggregates without medical narratives."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import requests
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from dmease.evidence_service import PublicEvidenceService
from dmease.types import ParsedSymptom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v2"


def normalize(label):
    label = re.sub(r"\s+", "", label)
    return label[:-1] if label.endswith("证") else label


def metrics(rows, graph_labels):
    if not rows:
        return {"records": 0}
    vocabulary = sorted({x for r in rows for x in r["gold"]} | graph_labels)
    mlb = MultiLabelBinarizer(classes=vocabulary)
    truth = mlb.fit_transform([r["gold"] for r in rows])
    prediction = mlb.transform([r["predicted"] for r in rows])
    gold_columns = [i for i in range(len(vocabulary)) if truth[:,i].any()]
    single = [r for r in rows if len(r["gold"]) == 1]
    gold_single = [r["gold"][0] for r in single]
    predicted_single = [r["predicted"][0] if r["predicted"] else "__abstain__" for r in single]
    return {"records": len(rows), "top1_any_gold_hit": float(np.mean([bool(r["predicted"]) and r["predicted"][0] in r["gold"] for r in rows])),
            "top3_any_gold_hit": float(np.mean([bool(set(r["gold"]) & set(r["predicted"])) for r in rows])),
            "top3_exact_set": float(np.mean([set(r["gold"]) == set(r["predicted"]) for r in rows])),
            "multilabel_micro_f1": float(f1_score(truth, prediction, average="micro", zero_division=0)),
            "macro_f1_gold_labels": float(f1_score(truth[:, gold_columns], prediction[:, gold_columns], average="macro", zero_division=0)),
            "abstentions": sum(not r["predicted"] for r in rows),
            "single_label_records": len(single),
            "single_label_top1_accuracy": float(np.mean(np.array(gold_single)==np.array(predicted_single))) if single else None,
            "single_label_macro_f1": float(f1_score(gold_single, predicted_single, labels=sorted(set(gold_single)), average="macro", zero_division=0)) if single else None,
            "single_label_kappa": float(cohen_kappa_score(gold_single, predicted_single)) if len(set(gold_single))>1 else None}


def main():
    protocol = json.loads((OUT / "external_protocol.json").read_text(encoding="utf8"))
    raw = ROOT / "data/raw/public/tcmeval_sdt/Train_TCM_Data_v1.json"
    if not raw.exists():
        response = requests.get(protocol["source_url"], timeout=45); response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != protocol["source_sha256"]:
            raise ValueError("External source version changed; freeze a new evaluation before using it")
        raw.parent.mkdir(parents=True, exist_ok=True);raw.write_bytes(response.content)
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == protocol["source_sha256"]
    records = json.loads(raw.read_text(encoding="utf8"))
    if len(records) != 200:
        raise ValueError("Unexpected record count")
    audit = []
    outputs = {method: [] for method in protocol["methods"]}
    with tempfile.TemporaryDirectory(prefix="dmease-external-") as temporary:
        service = PublicEvidenceService(ROOT / "data/processed/evidence_bundle.json", Path(temporary)/"patients.db", None)
        graph_labels = {normalize(s["name"]) for s in service.graph.syndromes}
        for record in records:
            text = record["Clinical Data"]
            gold = sorted({normalize(x) for x in re.split("[;；]", record["TCM Syndrome"]) if x.strip()})
            if not text.strip() or not gold:
                raise ValueError("Missing input/gold; do not silently change the denominator")
            digest = hashlib.sha256(text.encode()).hexdigest()
            parsed = service.parser.parse(text)
            literal = [ParsedSymptom(s["name"], s["name"], True, 1., "literal match") for s in service.graph.symptoms
                       if any(alias in text for alias in [s["name"], *s.get("aliases", [])])]
            metadata = {"row_hash": digest, "gold_in_graph": sum(g in graph_labels for g in gold),
                        "gold_count": len(gold), "explicit_gold_mention": any(g in re.sub(r"\s+", "", text) for g in gold),
                        "positive_parsed_symptoms": sum(s.present for s in parsed)}
            audit.append(metadata)
            for name, extracted in zip(protocol["methods"], (parsed, literal)):
                prediction = [normalize(s.name) for s in service.diagnose(extracted)]
                outputs[name].append({**metadata, "gold": gold, "predicted": prediction})
        fingerprint = service.evidence.fingerprint
    total_gold = sum(r["gold_count"] for r in audit)
    summaries = {}
    for name, rows in outputs.items():
        summaries[name] = {"all_records": metrics(rows, graph_labels),
                           "without_explicit_gold_in_input": metrics([r for r in rows if not r["explicit_gold_mention"]], graph_labels),
                           "at_least_one_gold_label_in_graph_secondary": metrics([r for r in rows if r["gold_in_graph"]], graph_labels)}
    sources = ["scripts/external_syndrome_audit.py", "src/dmease/evidence_service.py", "src/dmease/symptom_parser.py"]
    report = {"protocol": protocol, "evidence_fingerprint": fingerprint,
              "source_lf_sha256": {s: hashlib.sha256((ROOT/s).read_text(encoding="utf8").encode()).hexdigest() for s in sources},
              "completed_at": datetime.now(timezone.utc).isoformat(), "records": len(audit),
              "duplicate_text_excess": sum(n-1 for n in Counter(r["row_hash"] for r in audit).values()),
              "explicit_gold_mentions": sum(r["explicit_gold_mention"] for r in audit),
              "total_gold_labels": total_gold, "gold_labels_in_graph": sum(r["gold_in_graph"] for r in audit),
              "records_with_any_gold_in_graph": sum(r["gold_in_graph"]>0 for r in audit),
              "records_with_no_positive_symptom": sum(r["positive_parsed_symptoms"]==0 for r in audit),
              "evaluation": summaries}
    (OUT/"external_syndrome.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("protocol", "source_lf_sha256")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

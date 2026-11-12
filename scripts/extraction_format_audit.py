"""Separate post-hoc formatting audit of the same frozen model generations."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from dmease.evidence_service import PublicEvidenceService
from dmease.grounded_response import parse_grounded_response
from dmease.local_extraction import set_counts, summarize_counts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v3"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    primary = read(OUT/"extraction.json")
    source = ROOT/"data/raw/public/tcmeval_sdt/Train_TCM_Data_v1.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == primary["config"]["source_sha256"]
    records = read(source)
    assert len(records) == primary["records"] == 200
    rows, adjustments, failure_reasons = [], Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="dmease-format-audit-") as temp:
        service = PublicEvidenceService(ROOT/"data/processed/evidence_bundle.json", Path(temp)/"patient.db", None)
        assert service.evidence.fingerprint == primary["evidence_fingerprint"]
        names = {s["name"] for s in service.graph.symptoms}
        for i,record in enumerate(records):
            saved = read(ROOT/f"outputs/research_v3/extraction/{i:03d}.json")
            text = record["Clinical Data"]
            assert saved["row_hash"] == hashlib.sha256(text.encode()).hexdigest() == primary["rows"][i]["row_hash"]
            assert saved["stamp"] == primary["provenance"]
            if primary["rows"][i]["invalid_json"]:
                try:
                    json.loads(saved["raw"])
                    failure_reasons["valid_json_wrong_top_level_schema"] += 1
                except json.JSONDecodeError as e:
                    failure_reasons[e.msg] += 1
            parsed = parse_grounded_response(saved["raw"], text, names)
            adjustments[parsed["format_adjustment"]] += 1
            gold = {s.canonical for s in service.parser.parse(record["Clinical Information"]) if s.present}
            predicted = {r["canonical"] for r in parsed["items"] if r["present"] and r["canonical"] is not None}
            row = {"row_hash": saved["row_hash"], "counts": set_counts(gold,predicted), "adjustment": parsed["format_adjustment"],
                   "invalid_json": parsed["invalid_json"], "rejected": parsed["rejected"]}
            if row["adjustment"] in ("none", "unrecoverable"):
                assert row["counts"] == primary["rows"][i]["canonical"]["local_llm"]
            rows.append(row)
    sources = ("scripts/extraction_format_audit.py", "src/dmease/grounded_response.py", "src/dmease/local_extraction.py")
    result = {"protocol": read(OUT/"format_audit_protocol.json"),
              "source_lf_sha256": {name: hashlib.sha256((ROOT/name).read_text(encoding="utf8").encode()).hexdigest() for name in sources},
              "primary_provenance": primary["provenance"], "records": len(rows), "strict_json_failures": primary["quality"]["invalid_json"],
              "strict_failure_reasons": dict(failure_reasons), "format_adjustments": dict(adjustments),
              "remaining_json_failures": sum(r["invalid_json"] for r in rows), "summary": summarize_counts([r["counts"] for r in rows]), "rows": rows}
    (OUT/"extraction_format_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows", "source_lf_sha256", "primary_provenance", "protocol")},ensure_ascii=False))


if __name__ == "__main__":
    main()

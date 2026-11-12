"""Run the frozen local LLM once per public record, with resumable private cache."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dmease.evidence_service import PublicEvidenceService
from dmease.local_extraction import build_messages, validate_output, set_counts, summarize_counts
from dmease.types import ParsedSymptom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    config = protocol["extraction"]
    source = ROOT/"data/raw/public/tcmeval_sdt/Train_TCM_Data_v1.json"
    assert sha(source) == config["source_sha256"]
    records = json.loads(source.read_text(encoding="utf8"))
    assert len(records) == config["records"]
    hashes = {s: hashlib.sha256((ROOT/s).read_text(encoding="utf8").encode()).hexdigest()
              for s in ("scripts/local_extraction_benchmark.py", "src/dmease/local_extraction.py", "src/dmease/symptom_parser.py", "src/dmease/evidence_service.py")}
    stamp = {"protocol_sha256": sha(OUT/"protocol.json"), "source_lf_sha256": hashes}
    private = ROOT/"outputs/research_v3/extraction"; private.mkdir(parents=True, exist_ok=True)
    freeze = private/"freeze.json"
    if freeze.exists():
        assert json.loads(freeze.read_text(encoding="utf8")) == stamp, "Source changed after generation; do not mix runs"
    else:
        freeze.write_text(json.dumps(stamp, indent=2), encoding="utf8")
    local_model = ROOT/"outputs/research_v3/local_model"
    model_manifest = json.loads((local_model/"verified_model.json").read_text(encoding="utf8"))
    assert model_manifest["model"] == config["model"] and model_manifest["revision"] == config["revision"]
    with (local_model/"model.safetensors").open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == model_manifest["weights_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(str(local_model), local_files_only=True)
    model = None
    with tempfile.TemporaryDirectory(prefix="dmease-extract-") as temp:
        service = PublicEvidenceService(ROOT/"data/processed/evidence_bundle.json", Path(temp)/"patients.db", None)
        names = {s["name"] for s in service.graph.symptoms}
        rows, concepts = [], {method: {name: [] for name in names} for method in ("rules", "local_llm")}
        for i, record in enumerate(records):
            text = record[config["input_field"]]
            row_hash = hashlib.sha256(text.encode()).hexdigest()
            cache = private/f"{i:03d}.json"
            if cache.exists():
                generated = json.loads(cache.read_text(encoding="utf8"))
                assert generated["row_hash"] == row_hash and generated["stamp"] == stamp
            else:
                if model is None:
                    torch.manual_seed(42)
                    model = AutoModelForCausalLM.from_pretrained(str(local_model),
                              local_files_only=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
                messages = build_messages(text, service.graph.symptoms)
                inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
                original_tokens = inputs["input_ids"].shape[1]
                # Preserve the complete system instruction; truncate only the narrative if needed.
                was_truncated = False
                if original_tokens > config["max_input_tokens"]:
                    overhead = tokenizer.apply_chat_template(build_messages("", service.graph.symptoms), tokenize=True, add_generation_prompt=True)
                    budget = config["max_input_tokens"] - len(overhead) - 8
                    assert budget > 0
                    short = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:budget], skip_special_tokens=True)
                    inputs = tokenizer.apply_chat_template(build_messages(short, service.graph.symptoms), tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
                    was_truncated = True
                assert inputs["input_ids"].shape[1] <= config["max_input_tokens"]
                inputs = {k: v.to("cuda") for k,v in inputs.items()}
                torch.cuda.synchronize(); start = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                output = result[0, inputs["input_ids"].shape[1]:]
                generated = {"row_hash": row_hash, "stamp": stamp, "raw": tokenizer.decode(output, skip_special_tokens=True),
                             "input_truncated": was_truncated, "input_tokens": int(inputs["input_ids"].shape[1]),
                             "output_tokens": len(output), "output_limit": len(output) >= config["max_new_tokens"], "seconds": elapsed}
                cache.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf8")
            parsed = validate_output(generated["raw"], text, names)
            gold_terms = {re.sub(r"\s+", "", x) for x in re.split("[;；]", record[config["gold_field"]]) if x.strip()}
            gold = {x.canonical for x in service.parser.parse(record[config["gold_field"]]) if x.present}
            rules = service.parser.parse(text)
            predictions = {"rules": {x.canonical for x in rules if x.present},
                           "local_llm": {x["canonical"] for x in parsed["items"] if x["present"] and x["canonical"] is not None}}
            surface = {"rules": {re.sub(r"\s+", "", x.original) for x in rules if x.present},
                       "local_llm": {re.sub(r"\s+", "", x["term"]) for x in parsed["items"] if x["present"]}}
            normalize = lambda s: re.sub(r"\s+", "", s).removesuffix("证")
            syndromes = {normalize(x) for x in re.split("[;；]", record["TCM Syndrome"]) if x.strip()}
            downstream = {}
            for method, predicted in predictions.items():
                for name in names:
                    concepts[method][name].append(set_counts(gold & {name}, predicted & {name}))
                extracted = rules if method == "rules" else [ParsedSymptom(n, n, True, 1., "local model quotation validated") for n in sorted(predicted)]
                ranked = [normalize(x.name) for x in service.diagnose(extracted)]
                downstream[method] = {"top1": int(bool(ranked) and ranked[0] in syndromes), "top3": int(bool(set(ranked) & syndromes)), "abstain": int(not ranked)}
            row = {"row_hash": row_hash, "projected_reference_empty": not gold,
                   "canonical": {m: set_counts(gold, p) for m,p in predictions.items()},
                   "surface": {m: set_counts(gold_terms, p) for m,p in surface.items()}, "downstream": downstream,
                   "invalid_json": parsed["invalid_json"], "rejected": parsed["rejected"], "proposed_items": parsed["proposed_items"],
                   **{k: generated[k] for k in ("input_truncated", "input_tokens", "output_tokens", "output_limit", "seconds")}}
            rows.append(row)
            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{len(records)}; invalid JSON so far: {sum(r['invalid_json'] for r in rows)}", flush=True)
        summary = {}
        for method in predictions:
            summary[method] = {"canonical_all": summarize_counts([r["canonical"][method] for r in rows]),
                               "canonical_nonempty_reference": summarize_counts([r["canonical"][method] for r in rows if not r["projected_reference_empty"]]),
                               "strict_surface_diagnostic": summarize_counts([r["surface"][method] for r in rows]),
                               "downstream_exploratory": {k: sum(r["downstream"][method][k] for r in rows) / len(rows) for k in ("top1", "top3", "abstain")}}
        report = {"provenance": stamp, "completed_at": datetime.now(timezone.utc).isoformat(), "config": config,
                  "evidence_fingerprint": service.evidence.fingerprint, "records": len(rows), "summary": summary,
                  "per_concept": {m: {n: summarize_counts(v) for n,v in d.items()} for m,d in concepts.items()},
                  "quality": {"invalid_json": sum(r["invalid_json"] for r in rows), "input_truncated": sum(r["input_truncated"] for r in rows),
                              "output_limit": sum(r["output_limit"] for r in rows), "empty_projected_reference": sum(r["projected_reference_empty"] for r in rows),
                              "rejected_items": dict(sum((Counter(r["rejected"]) for r in rows), Counter())),
                              "median_seconds": float(np.median([r["seconds"] for r in rows])), "total_generation_seconds": sum(r["seconds"] for r in rows)},
                  "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}, "model_manifest": model_manifest, "rows": rows}
        (OUT/"extraction.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        print(json.dumps({"summary": summary, "quality": report["quality"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

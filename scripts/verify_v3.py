"""Offline numerical and provenance verification; no model or patient data needed."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v3"


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf8"))


def lf_sha(path):
    return hashlib.sha256(path.read_text(encoding="utf8").encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8), (a, b)


def json_file_matches(path, expected):
    # The execution host writes CRLF; Git archives normalize text to LF.
    # Accept only this transport difference, never changes to JSON contents.
    text = path.read_text(encoding="utf8")
    return expected in {hashlib.sha256(text.encode()).hexdigest(),
                        hashlib.sha256(text.replace("\n", "\r\n").encode()).hexdigest()}


def sources(hashes):
    for filename, expected in hashes.items():
        assert lf_sha(ROOT/filename) == expected, filename


def metrics(rows, method, key):
    counts = {k: sum(r[key][method][k] for r in rows) for k in ("tp", "fp", "fn", "exact", "gold_count", "predicted_count")}
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {**counts, "records": len(rows), "precision": tp/(tp+fp) if tp+fp else 0.,
            "recall": tp/(tp+fn) if tp+fn else 0., "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "exact_set_rate": counts["exact"]/len(rows) if rows else 0.}


def main():
    protocol, affinity, manifest = read("protocol.json"), read("affinity.json"), read("affinity_sources.json")
    sources(affinity["provenance"]["source_lf_sha256"])
    assert affinity["provenance"]["protocol_sha256"] == lf_sha(OUT/"protocol.json")
    assert json_file_matches(OUT/"affinity_sources.json", affinity["source_manifest_sha256"])
    assert manifest["curated_rows"] == len(affinity["row_metadata"])
    assert json_file_matches(OUT/"affinity_curated.json", manifest["curated_sha256"])
    curated = read("affinity_curated.json")
    assert len(curated) == manifest["curated_rows"]
    by_compound = {(r["target"],r["endpoint"],hashlib.sha256(r["smiles"].encode()).hexdigest()): r for r in curated}
    assignments = {}
    for row in affinity["row_metadata"]:
        previous = assignments.setdefault(row["scaffold_hash"], row["split"])
        assert previous == row["split"], "Scaffold leakage across strata"
        assert row["replicate_count"] == len(row["activity_ids"])
    for audit in manifest["audit"].values():
        assert audit["fetched"] == audit["eligible"] + sum(audit["excluded_first_reason"].values())
    for name, stratum in affinity["strata"].items():
        target, endpoint = name.split("_")
        subset = [r for r in affinity["row_metadata"] if r["target"] == target and r["endpoint"] == endpoint]
        assert len(subset) == sum(stratum["counts"].values())
        for split, count in stratum["counts"].items():
            assert sum(r["split"] == split for r in subset) == count
        if stratum["status"] != "evaluated":
            assert any(stratum["counts"][s] < n for s,n in protocol["affinity"]["minimum_counts"].items())
            continue
        selection = read(f"affinity_selection_{name}.json")
        assert selection == stratum["selection"]
        assert selection["provenance"] == affinity["provenance"]
        candidates = selection["candidates"]
        assert [c["config"] for c in candidates] == protocol["affinity"]["candidates"]
        selected = selection["selected_index"]
        assert selected == min(range(len(candidates)), key=lambda i: candidates[i]["validation"]["MAE_pK"])
        assert len(stratum["predictions"]) == stratum["counts"]["test"]
        for i, result in enumerate(stratum["test"]):
            for r in stratum["predictions"]:
                measured = by_compound[(target, endpoint, r["compound_hash"])]
                assert measured["split"] == "test"
                close(measured["pK"], r["pK"])
            delta = [r["predictions"][i]-r["pK"] for r in stratum["predictions"]]
            close(statistics.mean(abs(d) for d in delta), result["MAE_pK"])
            close(math.sqrt(statistics.mean(d*d for d in delta)), result["RMSE_pK"])
        close(stratum["test"][selected]["MAE_pK"]-stratum["test"][0]["MAE_pK"], stratum["comparison"]["MAE_selected_minus_mean"])
        ci = stratum["comparison"]["ci95"]
        assert len(ci) == 2 and ci[0] <= ci[1]
    initial_folder = OUT/"initial_affinity_implementation"
    old = json.loads((initial_folder/"affinity.json").read_text(encoding="utf8"))
    for name, checksum in old["provenance"]["source_lf_sha256"].items():
        assert lf_sha(initial_folder/Path(name).name.replace(".py", ".txt")) == checksum
    assert read("affinity_implementation_amendment.json")["prior_test_observed"] is True
    diagnostic = read("syndrome_diagnostics.json")
    sources(diagnostic["source_lf_sha256"])
    assert sum(r["test_records"] for r in diagnostic["frequency_bands"]) == 5486
    assert sum(r["labels"] for r in diagnostic["frequency_bands"]) == len(diagnostic["by_label"]) == 148
    close(statistics.mean(r["f1"] for r in diagnostic["by_label"]), .4246721430235834)
    transfer = diagnostic["transfer"]
    close(transfer["top1_known_label_ceiling"], transfer["records_with_any_known_gold"]/transfer["records"])
    close(transfer["overall_top1"], transfer["conditional_top1_known_gold"]*transfer["top1_known_label_ceiling"])
    imbalance, selection = read("imbalance.json"), read("imbalance_selection.json")
    assert imbalance["selection"] == selection
    sources(selection["provenance"]["source_lf_sha256"])
    assert selection["provenance"]["protocol"] == read("imbalance_protocol.json")
    assert selection["selected_index"] == max(range(len(selection["candidates"])), key=lambda i: selection["candidates"][i]["validation"]["macro_f1_148_train_labels"])
    assert len(selection["candidates"]) == 7 and selection["validation_records"] == 4957
    old = json.loads((ROOT/"experiments/v2/supervised_syndrome.json").read_text(encoding="utf8"))
    for method,result in imbalance["results"].items():
        close(statistics.mean(r["f1"] for r in result["by_label"]), result["official_test"]["macro_f1_148_train_labels"])
        assert sum(r["test"] for r in result["by_label"]) == result["official_test"]["records"] == 5486
        assert result["unseen_text_test"]["records"] == 4888
        assert sum(r["test_records"] for r in result["frequency_bands"]) == 5486
        if method == "frozen_unweighted_v2":
            for group in ("official_test", "unseen_text_test"):
                for key,value in result[group].items():
                    close(value, old["evaluation"]["supervised_text_baseline"][group][key])
    extraction = read("extraction.json")
    sources(extraction["provenance"]["source_lf_sha256"])
    assert extraction["provenance"]["protocol_sha256"] == lf_sha(OUT/"protocol.json")
    assert extraction["config"] == protocol["extraction"]
    assert len(extraction["rows"]) == extraction["records"] == 200
    assert len({r["row_hash"] for r in extraction["rows"]}) == 200
    for method, summary in extraction["summary"].items():
        for label, key, subset in (("canonical_all", "canonical", extraction["rows"]),
                ("canonical_nonempty_reference", "canonical", [r for r in extraction["rows"] if not r["projected_reference_empty"]]),
                ("strict_surface_diagnostic", "surface", extraction["rows"])):
            recomputed = metrics(subset, method, key)
            for k,value in recomputed.items():
                close(value, summary[label][k])
        for key in ("top1", "top3", "abstain"):
            close(statistics.mean(r["downstream"][method][key] for r in extraction["rows"]), summary["downstream_exploratory"][key])
        for k in ("tp", "fp", "fn"):
            assert sum(c[k] for c in extraction["per_concept"][method].values()) == summary["canonical_all"][k]
    for key in ("invalid_json", "input_truncated", "output_limit"):
        assert sum(r[key] for r in extraction["rows"]) == extraction["quality"][key]
    for row in extraction["rows"]:
        if row["invalid_json"]:
            assert row["canonical"]["local_llm"]["predicted_count"] == 0
    formatting = read("extraction_format_audit.json")
    sources(formatting["source_lf_sha256"])
    assert formatting["protocol"] == read("format_audit_protocol.json")
    assert formatting["primary_provenance"] == extraction["provenance"]
    assert len(formatting["rows"]) == formatting["records"] == 200
    assert formatting["strict_json_failures"] == extraction["quality"]["invalid_json"]
    assert sum(formatting["format_adjustments"].values()) == 200
    assert sum(formatting["strict_failure_reasons"].values()) == formatting["strict_json_failures"]
    assert sum(r["invalid_json"] for r in formatting["rows"]) == formatting["remaining_json_failures"]
    for row,primary in zip(formatting["rows"], extraction["rows"]):
        assert row["row_hash"] == primary["row_hash"]
        if row["adjustment"] in ("none", "unrecoverable"):
            assert row["counts"] == primary["canonical"]["local_llm"]
    rebuilt = metrics([{"counts": {"adapted": r["counts"]}} for r in formatting["rows"]], "adapted", "counts")
    for k,value in rebuilt.items():
        close(value,formatting["summary"][k])
    print("v3 provenance, all-scaffold isolation, selection, affinity metrics, classifier diagnostics and all 200 extraction denominators verified.")


if __name__ == "__main__":
    main()

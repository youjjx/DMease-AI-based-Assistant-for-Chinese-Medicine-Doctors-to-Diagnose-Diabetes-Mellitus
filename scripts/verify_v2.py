"""Verify published v2 experiments without raw medical data or model files."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v2"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf8"))


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)


def verify_sources(hashes):
    for name,expected in hashes.items():
        assert hashlib.sha256((ROOT/name).read_text(encoding="utf8").encode()).hexdigest()==expected,name


def main():
    summary, results = read("summary.json"), read("results.json")
    protocol, queries = read("protocol.json"), read("queries.json")["splits"]
    record=summary["provenance"]
    verify_sources(record["source_lf_sha256"])
    assert results["provenance"]==record
    assert record["protocol"]==hashlib.sha256(json.dumps(protocol,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    old=json.loads((ROOT/"experiments/queries.json").read_text(encoding="utf8"))["splits"]
    old_ids={r["id"] for rows in old.values() for r in rows}
    new_ids={r["id"] for rows in (queries["validation"],queries["test"]) for r in rows}
    assert len(new_ids)==288 and not new_ids & old_ids
    assert {r["query_id"] for r in read("training_teacher.json")["rows"]}=={r["id"] for r in queries["train"]}
    ids={r["id"] for r in queries["test"]}
    exact={r["query_id"]:r for r in results["exact"]}
    assert exact.keys()==ids
    for variant in protocol["variants"]:
        for seed in protocol["seeds"]:
            training=read(f"training/{variant}_{seed}.json")
            assert training["provenance"]==record
            chosen=next(h for h in training["validation_history"] if h["iteration"]==training["selected_iteration"])
            close(chosen["objective"],training["validation_objective"])
            assert max(h["objective"] for h in training["validation_history"])<=chosen["objective"]+1e-9
    for method,runs in results["methods"].items():
        for mode in ("raw","shared_local_search"):
            close(statistics.mean(r[mode]["mean"]["objective"] for r in runs),summary["results"][method][mode]["objective"])
            for run in runs:
                rows=run[mode]["rows"]
                assert len(rows)==192 and {r["query_id"] for r in rows}==ids
                close(statistics.mean(r["objective"] for r in rows),run[mode]["mean"]["objective"])
                for row in rows:
                    assert row["objective"]<=exact[row["query_id"]]["objective"]+1e-6
                    assert row["hard_constraint_violations"]==0
                    assert len(row["selected_herbs"])==len(set(row["selected_herbs"]))<=8
                    if row["union_all"]:
                        close(row["intersection_all"]/row["union_all"],row["target_jaccard_all_eligible"])
        for run in runs:
            for a,b in zip(run["raw"]["rows"],run["shared_local_search"]["rows"]):
                assert a["query_id"]==b["query_id"] and b["objective"]>=a["objective"]-1e-7
    external=read("external_syndrome.json")
    supervised=read("supervised_syndrome.json")
    verify_sources(external["source_lf_sha256"]);verify_sources(supervised["source_lf_sha256"])
    assert external["records"]==200 and supervised["train"]==43180 and supervised["test"]==5486
    selection=supervised["selection"]
    chosen=next(c for c in selection["candidates"] if c["configuration"]==selection["selected"])
    assert chosen["unseen_text_validation"]["macro_f1_148_train_labels"]==max(c["unseen_text_validation"]["macro_f1_148_train_labels"] for c in selection["candidates"])
    assert selection==read("syndrome_selection.json")
    assert set(selection["protocol"]["input_fields"]).isdisjoint(selection["protocol"]["forbidden_input_fields"])
    assert supervised["evaluation"]["supervised_text_baseline"]["unseen_text_test"]["records"]==5486-supervised["test_seen_texts_train_or_dev"]
    pathways=read("pathways.json")
    assert len(pathways["results"])==18
    for row in pathways["results"]:
        if row["status"]=="ok":
            request=pathways["requests"][row["request_id"]]
            assert row["pathways"]==request["pathways"]
            assert row["significant_pathways_fdr005"]==sum(float(t["fdr"])<.05 for t in row["pathways"])
            if row["background"]=="shared49":
                assert len(request["parameters"]["background_string_identifiers"].split("\r"))==len(pathways["background_mapped_proteins"])
    print("Verified 12 trained variants, new 96/192 disjoint queries, shared-search monotonicity, target Jaccard, pathway provenance, external and supervised syndrome reports.")


if __name__=="__main__":
    main()

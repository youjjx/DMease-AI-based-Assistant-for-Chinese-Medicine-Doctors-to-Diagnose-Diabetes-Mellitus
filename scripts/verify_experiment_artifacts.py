"""Verify published results and constraints without downloading raw databases."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9), (a, b)


def matches_text_digest(path, expected):
    # Git may translate LF/CRLF on checkout. Accept only that equivalence,
    # retaining the original run's raw-byte digest in the published records.
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return any(hashlib.sha256(value).hexdigest() == expected
               for value in (raw, lf, lf.replace(b"\n", b"\r\n")))


def main():
    summary = read(RESULTS / "summary.json")
    record = summary["provenance"]
    protocol = read(ROOT / "experiments/protocol.json")
    assert summary["protocol"] == protocol
    assert hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest() == record["protocol_sha256"]
    for path, expected in record["code_sha256"].items():
        assert matches_text_digest(ROOT / path, expected), path
    queries = read(ROOT / "experiments/queries.json")
    assert queries["evidence_fingerprint"] == record["evidence_fingerprint"]
    assert {k: len(v) for k, v in queries["splits"].items()} == summary["split_counts"]
    all_ids = [q["id"] for group in queries["splits"].values() for q in group]
    assert len(set(all_ids)) == len(all_ids)
    ids = {q["id"] for q in queries["splits"]["test"]}
    variants = {}
    for variant in protocol["variants"]:
        runs = [read(RESULTS / f"{variant}_{seed}.json") for seed in protocol["seeds"]]
        for run in runs:
            assert run["provenance"] == record
            assert {r["query_id"] for r in run["test"]["rows"]} == ids
            assert len(run["test"]["rows"]) == len(ids)
            for key, value in run["test"]["mean"].items():
                close(statistics.mean(r[key] for r in run["test"]["rows"]), value)
        for key in runs[0]["test"]["mean"]:
            values = [r["test"]["mean"][key] for r in runs]
            close(statistics.mean(values), summary["ablations"][variant]["seed_mean"][key])
            close(statistics.stdev(values), summary["ablations"][variant]["seed_std"][key])
        variants[variant] = runs
    optimal = read(RESULTS / "optimality.json")
    assert optimal["provenance"] == record
    exact = {r["query_id"]: r for r in optimal["exact"]}
    assert exact.keys() == ids
    baseline = {r["query_id"]: r for r in optimal["greedy"]["rows"]}
    close(statistics.mean(exact[q]["objective"]-baseline[q]["objective"] for q in ids), summary["exact_minus_greedy"]["mean_difference"])
    assert sum(r["enumerated_feasible_subsets"] for r in exact.values()) == summary["exact_subsets_evaluated"]
    for runs in variants.values():
        for run in runs:
            for row in run["test"]["rows"]:
                assert exact[row["query_id"]]["objective"] >= row["objective"]-1e-6
                assert len(row["selected_herbs"]) == len(set(row["selected_herbs"])) == row["herb_count"]
    constraints = read(RESULTS / "constraints.json")
    assert constraints["provenance"] == record
    total, violations = 0, 0
    for run in constraints["results"]:
        assert {r["query_id"] for r in run["rows"]} == ids
        for row in run["rows"]:
            chosen = row["selected_herbs"]
            first = baseline[row["query_id"]]["selected_herbs"]
            scenario = run["scenario"]
            assert len(chosen) == len(set(chosen))
            if scenario == "exclude_all":
                assert not chosen
            elif scenario == "exclude_first_baseline_choice":
                assert not set(first[:1]) & set(chosen)
            elif scenario == "only_one_allowed":
                assert len(chosen) <= 1
                if first:
                    assert not chosen or chosen == first[:1]
            elif len(first) >= 2:
                assert not set(first[:2]).issubset(chosen)
            assert row["hard_constraint_violations"] == 0
            total += 1
            violations += row["hard_constraint_violations"]
    assert total == summary["constraints_evaluated"] and violations == summary["constraint_violations"]
    for name, count in (("sensitivity", 14), ("missing_evidence", 18)):
        report = read(RESULTS / f"{name}.json")
        assert report["provenance"] == record and len(report["results"]) == count
        for run in report["results"]:
            assert {r["query_id"] for r in run["rows"]} == ids
            close(statistics.mean(r["objective"] for r in run["rows"]), run["mean"]["objective"])
    parser_after = read(RESULTS / "parser_after.json")
    parser_before = read(RESULTS / "parser_before.json")
    for report in (parser_before, parser_after):
        assert matches_text_digest(ROOT / "experiments/parser_cases.json", report["case_sha256"])
    assert matches_text_digest(ROOT / "src/dmease/symptom_parser.py", parser_after["parser_sha256"])
    for report in (parser_before, parser_after):
        for row in report["rows"]:
            assert row["exact"] == (row["predicted"] == row["expected"])
        assert sum(r["exact"] for r in report["rows"]) == report["summary"]["all"]["exact_matches"]
    print("Verified 21 training runs, 14 sensitivity settings, 18 evidence-drop runs, 1488 constraint executions, 62 exact optima, parser results and source hashes.")


if __name__ == "__main__":
    main()

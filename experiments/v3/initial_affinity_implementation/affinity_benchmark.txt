"""Fetch measured ChEMBL affinities and execute the preregistered scaffold split."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import joblib
import numpy as np
import requests
import rdkit
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from dmease.affinity_research import pk_from_nm, standardized_structure, scaffold_split, fingerprints, regression_metrics, scaffold_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v3"
RAW = ROOT/"data/raw/public/chembl_v3"
PRIVATE = ROOT/"outputs/research_v3/affinity"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached(resource, params, manifest):
    url = "https://www.ebi.ac.uk/chembl/api/data/" + resource + ".json"
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    path = RAW/f"{key}.json"
    if not path.exists():
        for attempt in range(4):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                payload = response.json()
                dump(path, payload)
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    meta = {"url": url, "params": params, "sha256": sha(path), "bytes": path.stat().st_size, "cache_file": path.name}
    if meta not in manifest:
        manifest.append(meta)
    return read(path)


def pages(resource, plural, params, manifest):
    rows, offset = [], 0
    while True:
        data = cached(resource, {**params, "limit": 1000, "offset": offset}, manifest)
        rows.extend(data[plural])
        if len(rows) >= data["page_meta"]["total_count"]:
            return rows
        if not data[plural]:
            raise ValueError("Incomplete API pagination")
        offset += len(data[plural])


def prepare(config):
    manifest = []
    status = cached("status", {}, manifest)
    grouped, audit = defaultdict(list), {}
    for name, target in config["targets"].items():
        info = cached("target/" + target, {}, manifest)
        assert info["organism"] == "Homo sapiens" and info["target_type"] == "SINGLE PROTEIN"
        activities = pages("activity", "activities", {"target_chembl_id": target, "standard_type__in": "Ki,Kd", "standard_relation": "=", "standard_units": "nM", "assay_type": "B"}, manifest)
        assays = pages("assay", "assays", {"target_chembl_id": target, "assay_type": "B", "confidence_score": 9}, manifest)
        allowed_assays = {r["assay_chembl_id"] for r in assays if r["confidence_score"] == 9 and r["target_chembl_id"] == target}
        excluded = Counter(); accepted = 0
        for row in activities:
            reason = None
            if row["assay_chembl_id"] not in allowed_assays:
                reason = "assay_confidence_not_9"
            elif row.get("data_validity_comment"):
                reason = "data_validity_comment"
            elif row.get("potential_duplicate"):
                reason = "potential_duplicate"
            elif row.get("standard_flag") != 1:
                reason = "nonstandard"
            elif row.get("assay_variant_mutation"):
                reason = "annotated_mutant"
            elif row.get("target_organism") != "Homo sapiens":
                reason = "organism"
            elif row.get("standard_type") not in config["endpoints"] or row.get("standard_units") != "nM" or row.get("standard_relation") != "=":
                reason = "endpoint_units_relation"
            if reason:
                excluded[reason] += 1
                continue
            try:
                value = pk_from_nm(float(row["standard_value"]))
            except (ValueError, TypeError):
                excluded["invalid_value"] += 1
                continue
            try:
                canonical, scaffold = standardized_structure(row["canonical_smiles"] or "")
                if not canonical:
                    raise ValueError("Empty structure")
            except (ValueError, TypeError, RuntimeError):
                excluded["invalid_structure"] += 1
                continue
            grouped[(name, row["standard_type"], canonical)].append({"pK": value, "scaffold": scaffold,
                "activity_id": row["activity_id"], "molecule_id": row["molecule_chembl_id"],
                "assay_id": row["assay_chembl_id"], "document_id": row["document_chembl_id"]})
            accepted += 1
        audit[name] = {"target": target, "pref_name": info["pref_name"], "components": [c["accession"] for c in info["target_components"]],
                       "fetched": len(activities), "eligible": accepted, "excluded_first_reason": dict(excluded)}
        print(f"{name}: fetched {len(activities)}, eligible {accepted}", flush=True)
    curated = []
    for (target, endpoint, smiles), rows in sorted(grouped.items()):
        values = [r["pK"] for r in rows]
        scaffold = rows[0]["scaffold"]
        curated.append({"target": target, "endpoint": endpoint, "smiles": smiles, "scaffold": scaffold,
                        "split": scaffold_split(scaffold), "pK": statistics.median(values), "replicate_count": len(rows),
                        "replicate_range": max(values)-min(values), "activity_ids": sorted({r["activity_id"] for r in rows}),
                        "molecule_ids": sorted({r["molecule_id"] for r in rows}), "assay_ids": sorted({r["assay_id"] for r in rows}),
                        "document_ids": sorted({r["document_id"] for r in rows if r["document_id"]})})
    dump(PRIVATE/"curated.json", curated)
    dump(OUT/"affinity_sources.json", {"api_status": status, "retrieved_at": datetime.now(timezone.utc).isoformat(), "requests": manifest,
                                      "audit": audit, "curated_sha256": sha(PRIVATE/"curated.json"), "curated_rows": len(curated)})
    return curated


def make_model(config):
    if config["method"] == "mean":
        return DummyRegressor(strategy="mean")
    if config["method"] == "ridge":
        return Ridge(alpha=config["alpha"], solver="lsqr", tol=1e-6)
    return RandomForestRegressor(n_estimators=config["trees"], min_samples_leaf=config["min_samples_leaf"],
                                 max_features=config["max_features"], random_state=42, n_jobs=2)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol = read(OUT/"protocol.json"); config = protocol["affinity"]
    sources = ("scripts/affinity_benchmark.py", "src/dmease/affinity_research.py")
    stamp = {"protocol_sha256": sha(OUT/"protocol.json"), "source_lf_sha256": {s: hashlib.sha256((ROOT/s).read_text(encoding="utf8").encode()).hexdigest() for s in sources}}
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if (PRIVATE/"curated.json").exists():
        assert sha(PRIVATE/"curated.json") == read(OUT/"affinity_sources.json")["curated_sha256"]
        curated = read(PRIVATE/"curated.json")
    else:
        curated = prepare(config)
    if args.prepare_only:
        return
    report = {"provenance": stamp, "environment": {"rdkit": rdkit.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
              "source_manifest_sha256": sha(OUT/"affinity_sources.json"), "strata": {}, "row_metadata": []}
    for target in config["targets"]:
        for endpoint in config["endpoints"]:
            key = f"{target}_{endpoint}"
            rows = [r for r in curated if r["target"] == target and r["endpoint"] == endpoint]
            idx = {s: [i for i,r in enumerate(rows) if r["split"] == s] for s in ("train", "validation", "test")}
            groups = {s: {rows[i]["scaffold"] for i in ix} for s,ix in idx.items()}
            docs = {s: {d for i in ix for d in rows[i]["document_ids"]} for s,ix in idx.items()}
            assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
            metadata = {"counts": {s: len(ix) for s,ix in idx.items()}, "scaffolds": {s: len(g) for s,g in groups.items()},
                        "document_overlap_train_test": len(docs["train"] & docs["test"]),
                        "replicate_conflicts_over_1_pK": sum(r["replicate_range"] > 1 for r in rows)}
            for row in rows:
                report["row_metadata"].append({"target": target, "endpoint": endpoint, "split": row["split"],
                    "compound_hash": hashlib.sha256(row["smiles"].encode()).hexdigest(), "scaffold_hash": hashlib.sha256(row["scaffold"].encode()).hexdigest(),
                    "molecule_ids": row["molecule_ids"], "activity_ids": row["activity_ids"], "assay_ids": row["assay_ids"], "document_ids": row["document_ids"],
                    "replicate_count": row["replicate_count"], "replicate_range": row["replicate_range"]})
            if any(len(idx[s]) < n for s,n in config["minimum_counts"].items()):
                report["strata"][key] = {**metadata, "status": "insufficient_fixed_split"}
                print(key, metadata["counts"], "insufficient; no resplitting", flush=True)
                continue
            x = fingerprints([r["smiles"] for r in rows], radius=config["features"]["radius"], bits=config["features"]["bits"])
            y = np.array([r["pK"] for r in rows])
            selection_path = OUT/f"affinity_selection_{key}.json"
            if selection_path.exists():
                selection = read(selection_path)
                assert selection["provenance"] == stamp and selection["curated_sha256"] == sha(PRIVATE/"curated.json")
                models = []
                for candidate in selection["candidates"]:
                    checkpoint = PRIVATE/candidate["checkpoint"]
                    assert sha(checkpoint) == candidate["checkpoint_sha256"]
                    models.append(joblib.load(checkpoint))  # Own local checksum-verified files only.
            else:
                candidates, models = [], []
                for i, candidate in enumerate(config["candidates"]):
                    model = make_model(candidate); model.fit(x[idx["train"]], y[idx["train"]])
                    model_path = PRIVATE/f"{key}_{i}.joblib"; joblib.dump(model, model_path)
                    metrics = regression_metrics(y[idx["validation"]], model.predict(x[idx["validation"]]))
                    candidates.append({"config": candidate, "validation": metrics, "checkpoint": model_path.name, "checkpoint_sha256": sha(model_path)})
                    models.append(model)
                selected = min(range(len(candidates)), key=lambda i: candidates[i]["validation"]["MAE_pK"])
                selection = {"provenance": stamp, "curated_sha256": sha(PRIVATE/"curated.json"), "selected_index": selected,
                             "selected_before_test_at": datetime.now(timezone.utc).isoformat(), "candidates": candidates}
                dump(selection_path, selection)
            # Only now compute test predictions; never update selection using test scores.
            test_predictions = [model.predict(x[idx["test"]]) for model in models]
            selected = selection["selected_index"]
            test = [regression_metrics(y[idx["test"]], p) for p in test_predictions]
            comparison = scaffold_bootstrap(y[idx["test"]], test_predictions[selected], test_predictions[0], [rows[i]["scaffold"] for i in idx["test"]])
            predictions = [{"compound_hash": hashlib.sha256(rows[ix]["smiles"].encode()).hexdigest(),
                            "scaffold_hash": hashlib.sha256(rows[ix]["scaffold"].encode()).hexdigest(), "pK": float(y[ix]),
                            "predictions": [float(p[j]) for p in test_predictions]} for j,ix in enumerate(idx["test"])]
            report["strata"][key] = {**metadata, "status": "evaluated", "selection": selection, "test": test,
                                     "comparison": comparison, "predictions": predictions}
            print(key, metadata["counts"], "selected", selected, "test MAE", test[selected]["MAE_pK"], flush=True)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    dump(OUT/"affinity.json", report)


if __name__ == "__main__":
    main()

"""Build a versioned recommendation graph from actual SymMap responses.

The existing UI lexicon only defines query scope and spelling aliases. None of
its symptom/syndrome/herb weights, mechanism edges or affinity numbers are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import requests


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    raw = root / "data/raw/public/symmap/relations"
    raw.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "data/processed/public_evidence.sqlite")
    def entities(kind):
        return [json.loads(row[0]) for row in db.execute(
            "SELECT payload FROM entities WHERE source='SymMap' AND version='2.0' AND kind=?", (kind,))]
    symptoms = {x["TCM_symptom_name"]: x for x in entities("SMTS") if not x.get("Suppress")}
    herbs = {x["Chinese_name"]: x for x in entities("SMHB") if not x.get("Suppress")}
    scope = json.loads((root / "data/knowledge_graph.json").read_text(encoding="utf8"))
    session = requests.Session()
    session.headers["User-Agent"] = "DMease-Research-Importer/2.0"
    files = []
    def related(rrid, kind):
        path = raw / f"{rrid}_{kind}.json"
        provenance = path.with_suffix(".provenance.json")
        url = "http://www.symmap.org/related_components/"
        params = {"rrid": rrid, "table_name": kind, "filter": "0"}
        cached = False
        if path.exists() and provenance.exists():
            meta = json.loads(provenance.read_text(encoding="utf8"))
            cached = meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() and meta["params"] == params
        if not cached:
            for attempt in range(3):
                try:
                    r = session.post(url, data=params, timeout=(10, 60))
                    r.raise_for_status()
                    data = r.json()
                    if "data" not in data or (data["data"] != "" and not isinstance(data["data"], list)):
                        raise ValueError("Unexpected SymMap response schema")
                    path.write_bytes(r.content)
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2)
            meta = {"url": url, "params": params, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "retrieved_at": datetime.now(timezone.utc).isoformat()}
            provenance.write_text(json.dumps(meta, indent=2), encoding="utf8")
            time.sleep(.25)
        data = json.loads(path.read_text(encoding="utf8"))
        files.append({**meta, "file": str(path.relative_to(root)).replace("\\", "/")})
        print(rrid, kind, len(data["data"] or []), flush=True)
        return data["data"] or []

    bundle = {"schema": "dmease-public-evidence-v1", "source": "SymMap 2.0 official relation responses / STRING 12.0",
              "data_scope": "public-database-associations", "symptoms": [], "herbs": [], "syndromes": {},
              "targets": {}, "symptom_herb": [], "symptom_syndrome": [], "symptom_target": [],
              "herb_target": [], "ppi": [], "unmatched": {"symptoms": [], "herbs": []}}
    for item in scope["herbs"]:
        row = herbs.get(item["name"])
        if not row:
            bundle["unmatched"]["herbs"].append(item["name"])
            continue
        bundle["herbs"].append({"id": f"SMHB{int(row['Herb_id']):05d}", "name": item["name"],
                                "source_url": f"http://www.symmap.org/detail/SMHB{int(row['Herb_id']):05d}"})
    herb_ids = {x["id"] for x in bundle["herbs"]}
    def targets(rows, source_id, relation):
        for row in rows:
            gene_id = row.get("Gene_id")
            if not gene_id:
                continue
            # Retain actual statistical metadata. FDR is not an affinity probability.
            fdr = row.get("FDR(BH)")
            eligible = isinstance(fdr, (float, int)) and 0 <= fdr < .05
            bundle["targets"][gene_id] = {"id": gene_id, "name": row.get("Gene_symbol", gene_id),
                                                "uniprot": row.get("UniProt_id"), "ensembl": row.get("Ensembl_id")}
            bundle[relation].append({"source": source_id, "target": gene_id, "eligible": eligible,
                                     "evidence": row, "source_url": f"http://www.symmap.org/detail/{source_id}"})
    for item in scope["symptoms"]:
        # Exact lexical match in a declared order; no fuzzy disease equivalence.
        match = next((name for name in [item["name"], *item.get("aliases", [])] if name in symptoms), None)
        if match is None:
            bundle["unmatched"]["symptoms"].append(item["name"])
            continue
        row = symptoms[match]
        rrid = f"SMTS{int(row['TCM_symptom_id']):05d}"
        bundle["symptoms"].append({"id": rrid, "name": match,
            "aliases": list(dict.fromkeys([item["name"], *item.get("aliases", [])])),
            "mapping": {"ui_term": item["name"], "matched_source_term": match, "method": "exact_name_or_declared_lexical_alias"},
            "source_url": f"http://www.symmap.org/detail/{rrid}"})
        for associated in related(rrid, "Herb"):
            if associated["Herb_id"] in herb_ids:
                bundle["symptom_herb"].append({"source": rrid, "target": associated["Herb_id"],
                    "evidence": associated, "source_url": f"http://www.symmap.org/detail/{rrid}"})
        for associated in related(rrid, "Syndrome"):
            sid = associated["Syndrome_id"]
            bundle["syndromes"][sid] = {"id": sid, "name": associated["Syndrome_name"]}
            bundle["symptom_syndrome"].append({"source": rrid, "target": sid, "evidence": associated,
                                               "source_url": f"http://www.symmap.org/detail/{rrid}"})
        targets(related(rrid, "Gene"), rrid, "symptom_target")
    for herb in bundle["herbs"]:
        targets(related(herb["id"], "Gene"), herb["id"], "herb_target")
    # Stable exact gene symbol mapping; ambiguous names are not merged.
    string_names = {}
    for protein_id, name in db.execute("SELECT id,name FROM entities WHERE source='STRING'"):
        string_names.setdefault(name, []).append(protein_id)
    mapping = {gid: string_names[x["name"]][0] for gid, x in bundle["targets"].items()
               if len(string_names.get(x["name"], [])) == 1}
    reverse = {protein: gene for gene, protein in mapping.items()}
    for a, b, score in db.execute("SELECT subject,object,score FROM edges WHERE source='STRING'"):
        if a in reverse and b in reverse:
            bundle["ppi"].append({"source": reverse[a], "target": reverse[b], "score": score,
                                  "string_ids": [a, b], "source_url": "https://version-12-0.string-db.org/"})
    db.close()
    bundle["string_mapping"] = mapping
    bundle["files"] = files
    bundle["limitations"] = ["Database associations include inferred evidence, not therapeutic efficacy or measured affinity",
                              "No clinical records or expert prescription labels", "Exact/declared lexical mapping; incomplete scope",
                              "Targets require recorded FDR(BH)<0.05; empty target evidence is retained as a gap"]
    for key in ("symptoms", "herbs"):
        bundle[key].sort(key=lambda x: x["id"])
    for key in ("symptom_herb", "symptom_syndrome", "symptom_target", "herb_target", "ppi"):
        bundle[key].sort(key=lambda x: (x["source"], x["target"]))
    # Retrieval timestamps are audit metadata, not model compatibility inputs.
    bundle["fingerprint"] = digest({k: v for k, v in bundle.items() if k != "files"})
    output = root / "data/processed/evidence_bundle.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(output)
    report = {"fingerprint": bundle["fingerprint"], "data_scope": bundle["data_scope"],
              "counts": {key: len(bundle[key]) for key in ("symptoms", "herbs", "syndromes", "targets", "symptom_herb", "symptom_syndrome", "symptom_target", "herb_target", "ppi")},
              "unmatched": bundle["unmatched"], "files": files, "limitations": bundle["limitations"]}
    (root / "data/evidence_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

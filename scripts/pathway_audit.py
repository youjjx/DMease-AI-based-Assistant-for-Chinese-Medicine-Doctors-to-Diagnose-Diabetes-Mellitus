"""Version-pinned STRING enrichment with fixed queries and explicit backgrounds."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
import torch

from dmease.evidence_policy import EvidenceGraph, canonical_hash
from dmease.experiment_metrics import graph_view, rollout
from dmease.kan import KANPolicy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v2"


def main():
    torch.set_num_threads(1)
    graph = EvidenceGraph(ROOT / "data/processed/evidence_bundle.json")
    protocol = json.loads((OUT / "pathway_protocol.json").read_text(encoding="utf8"))
    previous = json.loads((ROOT / "experiments/results/summary.json").read_text(encoding="utf8"))
    stamp = previous["provenance"]["protocol_sha256"][:12]
    cache = ROOT / "data/raw/public/string/pathway_audit"
    cache.mkdir(parents=True, exist_ok=True)
    mapping = graph.data["string_mapping"]
    universe = sorted({mapping[g] for g in graph.target_ids if g in mapping})
    session = requests.Session()
    requests_log, reports = {}, []
    for i, symptoms in enumerate(protocol["queries"]):
        if not set(symptoms).issubset(graph.snames):
            raise ValueError("Frozen query term is absent from current evidence")
        task = {"symptoms": dict.fromkeys(symptoms, 1.)}
        for method in protocol["methods"]:
            policy = None
            view = graph
            weight_hash = None
            if method != "greedy":
                variant = "full" if method == "v1_full_kan_seed42" else "without_ppi"
                path = ROOT / f"outputs/experiments/{stamp}/{variant}_42.pt"
                policy = KANPolicy(graph.state_dim, graph.action_dim)
                policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
                weight_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                policy.eval(); view = graph_view(graph, variant)
            actions, _ = rollout(view, task, policy, "model" if policy else "greedy")
            target_indices = np.flatnonzero(graph.affiliation[actions].sum(axis=0)>0) if actions else []
            target_ids = [graph.target_ids[j] for j in target_indices]
            proteins = sorted({mapping[t] for t in target_ids if t in mapping})
            for background in ("shared49", "human_exploratory"):
                row = {"case": i+1, "query": symptoms, "method": method, "model_sha256": weight_hash,
                       "background": background, "selected_herbs": [graph.herbs[j]["name"] for j in actions],
                       "targets_before_mapping": len(target_ids), "protein_count": len(proteins),
                       "unmapped_target_ids": [t for t in target_ids if t not in mapping]}
                if len(proteins)<2:
                    reports.append({**row, "status": "insufficient_targets", "pathways": []}); continue
                params = {"identifiers": "\r".join(proteins), "species": "9606", "caller_identity": "DMease_public_evidence_research"}
                if background == "shared49":
                    params["background_string_identifiers"] = "\r".join(universe)
                key = canonical_hash([protocol["source"], params])
                row["request_id"] = key
                if key not in requests_log:
                    local = cache / f"{key}.json"
                    audit_file = cache / f"{key}.audit.json"
                    try:
                        if local.exists() and audit_file.exists():
                            data = json.loads(local.read_text(encoding="utf8")); audit = json.loads(audit_file.read_text(encoding="utf8"))
                            if hashlib.sha256(local.read_bytes()).hexdigest() != audit["response_sha256"]:
                                raise ValueError("Cached response checksum mismatch")
                        else:
                            time.sleep(1.1)
                            response = session.get(protocol["source"], params=params, timeout=45)
                            response.raise_for_status(); data = response.json()
                            if not isinstance(data, list) or any("category" not in item for item in data):
                                raise ValueError("Unexpected STRING enrichment response")
                            local.write_bytes(response.content)
                            audit = {"url": protocol["source"], "parameters": params, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                     "response_sha256": hashlib.sha256(response.content).hexdigest(), "returned_terms": len(data)}
                            audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf8")
                        pathways = [item for item in data if item["category"] in protocol["categories"]]
                        requests_log[key] = {"status": "ok", **audit, "pathways": pathways}
                    except (requests.RequestException, ValueError) as error:
                        requests_log[key] = {"status": "failed", "error": str(error), "url": protocol["source"], "parameters": params, "pathways": []}
                entry = requests_log[key]
                reports.append({**row, "status": entry["status"], "pathways": entry["pathways"],
                                "significant_pathways_fdr005": sum(float(p["fdr"])<.05 for p in entry["pathways"])})
                print(f"Pathways case{i+1}/{method}/{background}: {entry['status']}, {len(entry['pathways'])} returned", flush=True)
    payload = {"protocol": protocol, "protocol_sha256": canonical_hash(protocol), "evidence_fingerprint": graph.fingerprint,
               "background_mapped_proteins": universe, "requests": requests_log, "results": reports}
    (OUT / "pathways.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()

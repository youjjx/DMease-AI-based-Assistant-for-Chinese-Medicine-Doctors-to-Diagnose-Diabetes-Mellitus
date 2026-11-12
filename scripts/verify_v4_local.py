"""Recompute every published planning score from the local licensed evidence graph."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dmease.evidence_policy import EvidenceGraph, EvidenceEnvironment
from dmease.experiment_metrics import exact_optimum

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'


def read(name):
    return json.loads((OUT/f'{name}.json').read_text(encoding='utf8'))


def main():
    torch.set_num_threads(1)
    graph = EvidenceGraph(ROOT/'data/processed/evidence_bundle.json')
    queries = read('planning_queries'); assert graph.fingerprint == queries['provenance']['evidence']
    lookup = {r['id']: r for r in queries['splits']['test']}
    envs = {}
    for key, task in lookup.items():
        env = EvidenceEnvironment(graph, max_steps=task['budget']); env.reset(task); envs[key] = env
    count = 0; maximum = 0.
    for filename in ('planning_results', 'gating_results'):
        for runs in read(filename)['methods'].values():
            for run in runs:
                for row in run['rows']:
                    chosen = np.zeros(graph.action_dim, dtype=np.float32)
                    chosen[[graph.hnames[h] for h in row['selected_herbs']]] = 1
                    measured = envs[row['query_id']].coverage(chosen)[2]
                    for recorded, key in (('objective', 'objective'), ('symptom_coverage', 'symptom_evidence_coverage'),
                                          ('target_coverage', 'target_evidence_coverage'), ('ppi_coverage', 'ppi_edge_coverage')):
                        error = abs(row[recorded]-measured[key]); maximum = max(maximum, error)
                        assert error < 1e-6, (filename, row['query_id'], recorded)
                    count += 1
    exact = read('planning_results')['methods']['exact'][0]['rows']; exact_error = 0.
    for row in exact:
        task = lookup[row['query_id']]
        reference = exact_optimum(graph, task, max_steps=task['budget'])
        error = abs(reference['objective']-row['objective']); exact_error = max(exact_error, error)
        assert error < 1e-6
    report = {'evidence_fingerprint': graph.fingerprint, 'recomputed_policy_rows': count, 'maximum_coverage_or_objective_error': maximum,
              'exact_reference_crosschecks': len(exact), 'maximum_exact_reference_error': exact_error,
              'source_lf_sha256': {'scripts/verify_v4_local.py': hashlib.sha256(Path(__file__).read_text(encoding='utf8').encode()).hexdigest()},
              'scope': 'All candidate sets rescored with the unchanged reference coverage implementation; all256 exact results crosschecked with the separate v2 enumeration implementation. Requires local evidence bundle.'}
    (OUT/'planning_score_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8', newline='\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

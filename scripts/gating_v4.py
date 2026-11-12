"""Validation-only margin selection and locked test of conservative overrides."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from dmease.evidence_policy import EvidenceGraph
from dmease.planning_v4 import PlanningEnvironment, PlanningPolicy
from dmease.conservative_planner import gated_rollout
from dmease.experiment_metrics import paired_statistics, holm_adjust
from planning_v4 import result_row, read, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'
PRIVATE = ROOT/'outputs/research_v4/planning'


def load(env, variant, seed, mode):
    meta = read(OUT/f'planning_training/{variant}_{seed}.json')
    path = PRIVATE/f'{variant}_{seed}{"_warm" if mode == "warm" else ""}.pt'
    key = 'warm_checkpoint_sha256' if mode == 'warm' else 'checkpoint_sha256'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta[key]
    policy = PlanningPolicy(env.observation_dim, env.feature_dim, variant)
    policy.load_state_dict(torch.load(path, weights_only=True, map_location='cpu')['state_dict']); policy.eval()
    return policy, meta[key]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--select', action='store_true'); parser.add_argument('--test', action='store_true')
    args = parser.parse_args(); assert args.select != args.test
    torch.set_num_threads(1)
    protocol = read(OUT/'gating_protocol.json'); queries = read(OUT/'planning_queries.json')
    graph = EvidenceGraph(ROOT/'data/processed/evidence_bundle.json'); assert graph.fingerprint == queries['provenance']['evidence']
    stamp = {'protocol': protocol, 'planning_provenance': queries['provenance'], 'source_lf_sha256': {
        p: hashlib.sha256((ROOT/p).read_text(encoding='utf8').encode()).hexdigest()
        for p in ('scripts/gating_v4.py', 'src/dmease/conservative_planner.py')}}
    # All candidate checkpoints must exist before calibration or testing begins.
    probe = PlanningEnvironment(graph, queries['splits']['validation'][0])
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            for mode in ('warm', 'selected'):
                load(probe, variant, seed, mode)
    if args.select:
        path = OUT/'gating_selection.json'
        if path.exists():
            assert read(path)['provenance'] == stamp
            print('Frozen gating selection already exists.'); return
        envs = [PlanningEnvironment(graph, t) for t in queries['splits']['validation']]
        baseline = float(np.mean([gated_rollout(env)[1] for env in envs])); selections = {}
        for variant in protocol['variants']:
            for seed in protocol['seeds']:
                candidates = [{'mode': 'greedy', 'margin': None, 'validation_objective': baseline}]
                best = baseline; chosen = 0
                for mode in ('warm', 'selected'):
                    policy, checkpoint_hash = load(probe, variant, seed, mode)
                    for margin in (0., .002, .005, .01, .02, .04, .08):
                        score = float(np.mean([gated_rollout(env, policy, margin)[1] for env in envs]))
                        candidates.append({'mode': mode, 'margin': margin, 'validation_objective': score, 'checkpoint_sha256': checkpoint_hash})
                        if score > best+1e-8:
                            best = score; chosen = len(candidates)-1
                selections[f'{variant}_{seed}'] = {'selected_index': chosen, 'candidates': candidates}
                print(variant, seed, 'selected', candidates[chosen], flush=True)
        save(path, {'provenance': stamp, 'selections': selections, 'frozen_before_test_at': datetime.now(timezone.utc).isoformat()})
        return
    selection = read(OUT/'gating_selection.json'); assert selection['provenance'] == stamp
    original = read(OUT/'planning_results.json')
    assert selection['frozen_before_test_at'] < original['completed_at']
    envs = [PlanningEnvironment(graph, t) for t in queries['splits']['test']]
    greedy_rows = original['methods']['greedy'][0]['rows']; methods = {}; summary = {}; comparisons = {}
    baseline_base = [{'query_id': base, 'objective': float(np.mean([r['objective'] for r in greedy_rows if r['base_id'] == base]))}
                     for base in sorted({r['base_id'] for r in greedy_rows})]
    for variant in protocol['variants']:
        methods[variant] = []; methods[variant+'_guarded'] = []
        for seed in protocol['seeds']:
            decision = selection['selections'][f'{variant}_{seed}']; selected = decision['candidates'][decision['selected_index']]
            policy = None if selected['mode'] == 'greedy' else load(probe, variant, seed, selected['mode'])[0]
            rows = []; guarded = []
            for env, baseline in zip(envs, greedy_rows):
                env.score.cache_clear(); start = time.perf_counter(); mask, _, changes = gated_rollout(env, policy, selected['margin'])
                elapsed = time.perf_counter()-start; row = {**result_row(env, mask, elapsed), **changes}; rows.append(row)
                prior = sum(1 << graph.hnames[h] for h in baseline['selected_herbs'])
                best = mask if row['objective'] > baseline['objective']+1e-7 else prior
                guarded.append(result_row(env, best, elapsed+baseline['seconds']))
            methods[variant].append({'seed': seed, 'selection': selected, 'rows': rows})
            methods[variant+'_guarded'].append({'seed': seed, 'rows': guarded})
        base_rows = [{'query_id': base, 'objective': float(np.mean([r['objective'] for run in methods[variant] for r in run['rows'] if r['base_id'] == base]))}
                     for base in sorted({r['base_id'] for r in greedy_rows})]
        comparisons[variant] = paired_statistics(base_rows, baseline_base)
    adjusted = holm_adjust([original['raw_vs_greedy_base_clustered'][v]['sign_flip_p'] for v in protocol['variants']]+[comparisons[v]['sign_flip_p'] for v in protocol['variants']])
    for variant, p in zip(protocol['variants'], adjusted[3:]):
        comparisons[variant]['holm_p_six_comparisons'] = p
    for method, runs in methods.items():
        summary[method] = {}
        for profile in ('all', 'standard', 'requested_constraints'):
            rows = [r for run in runs for r in run['rows'] if profile == 'all' or r['profile'] == profile]
            summary[method][profile] = {'rows_with_seeds': len(rows), 'objective': float(np.mean([r['objective'] for r in rows])),
                                        'hard_violations': sum(r['hard_violations'] for r in rows),
                                        'overrides': sum(r['overrides'] for r in rows) if all('overrides' in r for r in rows) else None,
                                        'median_ms': float(np.median([r['seconds'] for r in rows]))*1000}
    result = {'provenance': stamp, 'selection': selection, 'methods': methods, 'summary': summary, 'comparisons': comparisons,
              'raw_holm_p_six_comparisons': dict(zip(protocol['variants'], adjusted[:3])), 'completed_at': datetime.now(timezone.utc).isoformat()}
    save(OUT/'gating_results.json', result); save(OUT/'gating_summary.json', {k: v for k, v in result.items() if k != 'methods'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

"""Verify released v4 denominators, frozen selections, constraints and statistics offline."""
import hashlib
import json
from pathlib import Path

import numpy as np

from dmease.evidence_policy import canonical_hash
from dmease.experiment_metrics import paired_statistics, holm_adjust
from dmease.structured_extraction import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-9), (a, b)


def sources(stamp):
    for name, expected in stamp['source_lf_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf8').encode()).hexdigest() == expected, name


def main():
    protocol = read(OUT/'planning_protocol.json')
    queries = read(OUT/'planning_queries.json'); stamp = queries['provenance']; sources(stamp)
    assert stamp['protocol'] == canonical_hash(protocol)
    old = {canonical_hash(sorted(t['symptoms'])) for file in ('experiments/queries.json', 'experiments/v2/queries.json')
           for tasks in read(ROOT/file)['splits'].values() for t in tasks}
    ids = {}; base_sets = {}
    for name, count in (('train', 640), ('validation', 128), ('test', 256)):
        tasks = queries['splits'][name]; assert len(tasks) == count
        ids[name] = {t['id'] for t in tasks}; assert len(ids[name]) == count
        base_sets[name] = {t['base_id'] for t in tasks}; assert len(base_sets[name]) == count//2
        assert not base_sets[name] & old
        for task in tasks:
            assert canonical_hash({k: v for k, v in task.items() if k != 'id'}) == task['id']
            assert canonical_hash(sorted(task['symptoms'])) == task['base_id']
            assert task['profile'] in ('standard', 'requested_constraints')
    assert not any(base_sets[a] & base_sets[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))
    teacher = read(OUT/'planning_teacher.json'); assert teacher['provenance'] == stamp
    assert {r['query_id'] for r in teacher['rows']} == ids['train']
    assert sum(r['states'] for r in teacher['rows']) == teacher['states']
    result = read(OUT/'planning_results.json'); assert result['provenance'] == stamp
    frozen = []
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            training = read(OUT/f'planning_training/{variant}_{seed}.json'); assert training['provenance'] == stamp
            best = float('-inf'); selected = None
            for row in training['history']:
                if row['objective'] > best+1e-8:
                    best = row['objective']; selected = row['iteration']
            assert training['selected_iteration'] == selected
            close(best, training['validation_objective'])
            assert len(training['ppo_diagnostics']) == protocol['training']['ppo_iterations']
            checkpoint = ROOT/f'models/planning_v4/{variant}_{seed}.pt'
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == training['checkpoint_sha256']
            warm = ROOT/f'models/planning_v4/{variant}_{seed}_warm.pt'
            assert hashlib.sha256(warm.read_bytes()).hexdigest() == training['warm_checkpoint_sha256']
            frozen.append(training['frozen_before_test_at'])
    assert max(frozen) < result['completed_at']
    lookup = {t['id']: t for t in queries['splits']['test']}
    optimum = {r['query_id']: r['objective'] for r in result['methods']['exact'][0]['rows']}
    paired = {}
    for name, runs in result['methods'].items():
        assert len(runs) == (1 if name in ('greedy', 'lookahead2', 'greedy_search', 'exact') else 3)
        for run in runs:
            assert len(run['rows']) == len({r['query_id'] for r in run['rows']}) == 256
            assert {r['query_id'] for r in run['rows']} == ids['test']
            for row in run['rows']:
                task = lookup[row['query_id']]; herbs = set(row['selected_herbs'])
                assert row['base_id'] == task['base_id'] and row['profile'] == task['profile']
                assert len(herbs) == row['herb_count'] == len(row['selected_herbs']) <= task['budget']
                assert not herbs & set(task['forbidden'])
                assert not any(a in herbs and b in herbs for a, b in task['forbidden_pairs'])
                assert row['hard_violations'] == 0 and row['objective'] <= optimum[row['query_id']]+1e-6
                assert row['seconds'] >= 0
        for profile in ('all', 'standard', 'requested_constraints'):
            rows = [r for run in runs for r in run['rows'] if profile == 'all' or r['profile'] == profile]
            s = result['summary'][name][profile]
            assert len(rows) == s['evaluated_rows_with_seeds']
            close(np.mean([r['objective'] for r in rows]), s['objective'])
            close(np.mean([optimum[r['query_id']]-r['objective'] for r in rows]), s['mean_gap_to_optimal'])
            close(np.mean([abs(optimum[r['query_id']]-r['objective']) < 1e-6 for r in rows]), s['optimal_rate'])
        paired[name] = [{'query_id': base, 'objective': float(np.mean([r['objective'] for run in runs for r in run['rows'] if r['base_id'] == base]))}
                        for base in sorted(base_sets['test'])]
    comparisons = [paired_statistics(paired[v], paired['greedy']) for v in protocol['variants']]
    adjusted = holm_adjust([r['sign_flip_p'] for r in comparisons])
    for variant, stats, adj in zip(protocol['variants'], comparisons, adjusted):
        assert {**stats, 'holm_p': adj} == result['raw_vs_greedy_base_clustered'][variant]
        assert paired_statistics(paired[variant], paired[variant+'_warm']) == result['warm_to_selected'][variant]
    assert read(OUT/'planning_summary.json') == {k: v for k, v in result.items() if k != 'methods'}
    gated = read(OUT/'gating_results.json'); gate_selection = read(OUT/'gating_selection.json')
    sources(gated['provenance']); assert gated['provenance']['protocol'] == read(OUT/'gating_protocol.json')
    assert gated['selection'] == gate_selection
    assert gate_selection['frozen_before_test_at'] < result['completed_at'] < gated['completed_at']
    for record in gate_selection['selections'].values():
        assert len(record['candidates']) == 15
        best = float('-inf'); selected = 0
        for i, candidate in enumerate(record['candidates']):
            if candidate['validation_objective'] > best+1e-8:
                best = candidate['validation_objective']; selected = i
        assert record['selected_index'] == selected
    greedy_by_id = {r['query_id']: r for r in result['methods']['greedy'][0]['rows']}
    gated_pairs = {}
    for method, runs in gated['methods'].items():
        assert len(runs) == 3
        for run in runs:
            assert len(run['rows']) == 256 and {r['query_id'] for r in run['rows']} == ids['test']
            for row in run['rows']:
                task = lookup[row['query_id']]; herbs = set(row['selected_herbs'])
                assert len(herbs) == row['herb_count'] <= task['budget']
                assert not herbs & set(task['forbidden']) and not any(a in herbs and b in herbs for a, b in task['forbidden_pairs'])
                assert row['hard_violations'] == 0 and row['objective'] <= optimum[row['query_id']]+1e-6
                if method.endswith('_guarded'):
                    assert row['objective'] >= greedy_by_id[row['query_id']]['objective']-1e-7
        for profile in ('all', 'standard', 'requested_constraints'):
            rows = [r for run in runs for r in run['rows'] if profile == 'all' or r['profile'] == profile]
            close(np.mean([r['objective'] for r in rows]), gated['summary'][method][profile]['objective'])
        if not method.endswith('_guarded'):
            base_rows = [{'query_id': base, 'objective': float(np.mean([r['objective'] for run in runs for r in run['rows'] if r['base_id'] == base]))}
                         for base in sorted(base_sets['test'])]
            gated_pairs[method] = paired_statistics(base_rows, paired['greedy'])
    six_p = holm_adjust([result['raw_vs_greedy_base_clustered'][v]['sign_flip_p'] for v in protocol['variants']]+
                        [gated_pairs[v]['sign_flip_p'] for v in protocol['variants']])
    for i, variant in enumerate(protocol['variants']):
        assert {**gated_pairs[variant], 'holm_p_six_comparisons': six_p[i+3]} == gated['comparisons'][variant]
        close(six_p[i], gated['raw_holm_p_six_comparisons'][variant])
    assert read(OUT/'gating_summary.json') == {k: v for k, v in gated.items() if k != 'methods'}
    score_check = read(OUT/'planning_score_verification.json'); sources(score_check)
    assert score_check['evidence_fingerprint'] == stamp['evidence']
    expected_rows = sum(len(run['rows']) for report in (result, gated) for runs in report['methods'].values() for run in runs)
    assert score_check['recomputed_policy_rows'] == expected_rows and score_check['exact_reference_crosschecks'] == 256
    assert score_check['maximum_coverage_or_objective_error'] < 1e-6 and score_check['maximum_exact_reference_error'] < 1e-6
    diagnostics = read(OUT/'planning_diagnostics.json'); sources(diagnostics)
    assert diagnostics['planning_provenance'] == stamp
    for split, methods in diagnostics['results'].items():
        assert len(methods) == 19
        for row in methods.values():
            assert row['greedy_optimal_first_actions']+row['greedy_suboptimal_first_actions'] == row['states']
            assert row['damaged_greedy_optimal_actions'] <= row['greedy_optimal_first_actions']
            assert row['rescued_greedy_suboptimal_actions'] <= row['greedy_suboptimal_first_actions']
            if split == 'train':
                assert row['states'] == teacher['states']
    selection = read(OUT/'extraction_selection.json'); sources(selection['provenance'])
    assert selection['provenance']['protocol'] == read(OUT/'extraction_protocol.json')
    manifest = read(OUT/'extraction_sources.json')
    assert not any(manifest['normalized_text_overlap'].values())
    assert manifest['counts'] == {'train': 200, 'validation': 50, 'test': 50}
    for phase in ('validation', 'test'):
        extraction = read(OUT/f'extraction_{phase}.json')
        assert extraction['provenance'] == selection['provenance']
        if phase == 'test':
            assert selection['frozen_before_test_at'] < extraction['completed_at']
        hashes = None
        for method, rows in extraction['methods'].items():
            assert len(rows) == len({r['row_hash'] for r in rows}) == 50
            if hashes is None:
                hashes = {r['row_hash'] for r in rows}
            assert {r['row_hash'] for r in rows} == hashes
            summary = extraction['summary'][method]
            for metric in ('surface', 'characters', 'projected_secondary'):
                assert summarize([r[metric] for r in rows]) == summary[metric]
            for metric in ('invalid_json_chunks', 'invalid_schema_chunks', 'output_limits', 'chunks'):
                assert sum(r.get(metric, 0) for r in rows) == summary[metric]
            close(sum(r['locatable_gold'] for r in rows)/sum(r['total_gold'] for r in rows), summary['gold_verbatim_fraction'])
        if phase == 'validation':
            chosen = max(selection['provenance']['protocol']['variants'], key=lambda v: extraction['summary'][v]['surface']['f1'])
            assert chosen == selection['selected']
    diagnosis = read(OUT/'diagnosis_test.json'); sources(diagnosis)
    statistics = read(OUT/'extraction_statistics.json'); sources(statistics)
    from extraction_statistics_v4 import compare
    extraction = read(OUT/'extraction_test.json')
    stats = {method: compare(extraction['methods'][selection['selected']], extraction['methods'][method])
             for method in ('symmap_literal', 'copy_clauses', 'v3_strict')}
    adjusted = holm_adjust([s['paired_randomization_p'] for s in stats.values()])
    for (method, value), p in zip(stats.items(), adjusted):
        assert {**value, 'holm_p': p} == statistics['comparisons'][method]
    assert diagnosis['protocol'] == read(OUT/'diagnosis_protocol.json')
    for result in diagnosis['results'].values():
        rows = result['rows']; assert len(rows) == 50
        close(np.mean([r['top1'] for r in rows]), result['groups']['all']['top1'])
        close(np.mean([r['top3'] for r in rows]), result['groups']['all']['top3'])
    affinity = read(OUT/'affinity_robustness.json'); sources(affinity['provenance'])
    assert affinity['provenance']['protocol'] == read(OUT/'affinity_protocol.json')
    for result in affinity['groupings'].values():
        assignment = result['row_assignment']; assert len(assignment)+affinity['excluded_missing_document'] == 468
        groups = {s: {r['component'] for r in assignment if r['split'] == s} for s in ('train', 'validation', 'test')}
        assert not any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))
        for name, stratum in result['strata'].items():
            target, endpoint = name.split('_')
            for split, count in stratum['counts'].items():
                assert count == sum(r['target'] == target and r['endpoint'] == endpoint and r['split'] == split for r in assignment)
            if stratum['status'] == 'insufficient_fixed_split':
                assert any(stratum['counts'][s] < n for s, n in affinity['provenance']['protocol']['minimum_counts'].items())
    loco = read(OUT/'affinity_loco.json'); sources(loco['provenance'])
    assert loco['provenance']['protocol'] == read(OUT/'affinity_loco_protocol.json')
    for strata in loco['groupings'].values():
        for result in strata.values():
            folds = result['folds']; predictions = result['predictions']
            assert len(folds) == result['total_components']
            assert sum(f['test'] for f in folds) == result['rows']
            assert len(predictions) == result['evaluated_rows'] == len({r['row_index'] for r in predictions})
            close(result['coverage'], len(predictions)/result['rows'])
            assert all(f['train']+f['test'] == result['rows'] for f in folds)
            assert all((f['train'] >= 40) == (f['status'] == 'evaluated') for f in folds)
            if result['status'] == 'evaluated':
                for method, key in (('training_mean', 'mean'), ('ridge_alpha10', 'ridge')):
                    close(np.mean([abs(r['pK']-r[key]) for r in predictions]), result['pooled'][method]['MAE_pK'])
                close(result['pooled']['ridge_alpha10']['MAE_pK']-result['pooled']['training_mean']['MAE_pK'], result['comparison']['MAE_selected_minus_mean'])
    print('v4 source hashes, 9 frozen models, all partitions, 256-query constraints/statistics, 50-case denominators and robustness counts verified.')


if __name__ == '__main__':
    main()

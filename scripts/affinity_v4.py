"""Publication-disjoint robustness audit; fixed grouping without label-dependent splits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from affinity_benchmark import make_model
from dmease.affinity_research import fingerprints, regression_metrics, scaffold_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'
PRIVATE = ROOT/'outputs/research_v4/affinity'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8', newline='\n')


def grouping(rows, include_scaffold):
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    previous = {}
    for i, row in enumerate(rows):
        keys = [('doc', d) for d in row['document_ids']] + [('smiles', row['smiles'])]
        if include_scaffold:
            keys.append(('scaffold', row['scaffold']))
        for key in keys:
            if key in previous:
                parents[root(i)] = root(previous[key])
            else:
                previous[key] = i
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(root(i), set()).add(row['smiles'])
    names = {key: hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest() for key, values in members.items()}
    return [names[root(i)] for i in range(len(rows))]


def split(group):
    value = int(hashlib.sha256(('dmease-affinity-v4|'+group).encode()).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else 'validation' if value < 85 else 'test'


def main():
    protocol = read(OUT/'affinity_protocol.json')
    snapshot = ROOT/'experiments/v3/affinity_curated.json'
    text = snapshot.read_text(encoding='utf8')
    expected = read(ROOT/'experiments/v3/affinity_sources.json')['curated_sha256']
    assert expected in [hashlib.sha256(x.encode()).hexdigest() for x in (text, text.replace('\n', '\r\n'))]
    all_rows = json.loads(text); rows = [r for r in all_rows if r['document_ids']]
    stamp = {'protocol': protocol, 'snapshot_original_sha256': expected, 'source_lf_sha256': {
        p: hashlib.sha256((ROOT/p).read_text(encoding='utf8').encode()).hexdigest()
        for p in ('scripts/affinity_v4.py', 'scripts/affinity_benchmark.py', 'src/dmease/affinity_research.py')}}
    report = {'provenance': stamp, 'excluded_missing_document': len(all_rows)-len(rows), 'groupings': {}}
    for method in protocol['groupings']:
        groups = grouping(rows, method == 'document_and_scaffold')
        labels = [split(g) for g in groups]
        result = {'connected_components': len(set(groups)), 'largest_component_rows': max(groups.count(g) for g in set(groups)),
                  'strata': {}, 'row_assignment': [{'compound_hash': hashlib.sha256(r['smiles'].encode()).hexdigest(),
                    'target': r['target'], 'endpoint': r['endpoint'], 'component': g, 'split': s} for r, g, s in zip(rows, groups, labels)]}
        for target, endpoint in sorted({(r['target'], r['endpoint']) for r in rows}):
            key = target+'_'+endpoint
            indices = {s: [i for i, r in enumerate(rows) if r['target'] == target and r['endpoint'] == endpoint and labels[i] == s]
                       for s in ('train', 'validation', 'test')}
            docs = {s: {d for i in ids for d in rows[i]['document_ids']} for s, ids in indices.items()}
            scaffolds = {s: {rows[i]['scaffold'] for i in ids} for s, ids in indices.items()}
            assert not any(docs[a] & docs[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))
            if method == 'document_and_scaffold':
                assert not any(scaffolds[a] & scaffolds[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))
            meta = {'counts': {s: len(ids) for s, ids in indices.items()}, 'document_overlap_train_test': 0,
                    'scaffold_overlap_train_test': len(scaffolds['train'] & scaffolds['test']),
                    'components': {s: len({groups[i] for i in ids}) for s, ids in indices.items()}}
            if any(len(indices[s]) < n for s, n in protocol['minimum_counts'].items()):
                result['strata'][key] = {**meta, 'status': 'insufficient_fixed_split'}
                continue
            x = fingerprints([r['smiles'] for r in rows], radius=2, bits=2048)
            y = np.array([r['pK'] for r in rows]); candidates = []; models = []
            for j, config in enumerate(protocol['candidates']):
                model = make_model(config); model.fit(x[indices['train']], y[indices['train']])
                path = PRIVATE/f'{method}_{key}_{j}.joblib'; path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
                candidates.append({'config': config, 'validation': regression_metrics(y[indices['validation']], model.predict(x[indices['validation']])),
                                   'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                models.append(model)
            selected = min(range(len(candidates)), key=lambda j: candidates[j]['validation']['MAE_pK'])
            selection = {'provenance': stamp, 'candidates': candidates, 'selected_index': selected, 'frozen_before_test_at': datetime.now(timezone.utc).isoformat()}
            save(OUT/f'affinity_selection_{method}_{key}.json', selection)
            predictions = [m.predict(x[indices['test']]) for m in models]
            result['strata'][key] = {**meta, 'status': 'evaluated', 'selection': selection,
                                    'test': [regression_metrics(y[indices['test']], p) for p in predictions],
                                    'comparison': scaffold_bootstrap(y[indices['test']], predictions[selected], predictions[0], [groups[i] for i in indices['test']]),
                                    'predictions': [{'row_index': i, 'pK': float(y[i]), 'component': groups[i], 'values': [float(p[j]) for p in predictions]}
                                                    for j, i in enumerate(indices['test'])]}
        report['groupings'][method] = result
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    save(OUT/'affinity_robustness.json', report)
    print(json.dumps({k: {s: {'counts': v['counts'], 'status': v['status']} for s, v in r['strata'].items()} for k, r in report['groupings'].items()}), flush=True)


if __name__ == '__main__':
    main()

"""Leave-one-source-component-out audit with fixed previously chosen hyperparameters."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from affinity_v4 import grouping, read, save
from dmease.affinity_research import fingerprints, regression_metrics, scaffold_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'


def main():
    protocol = read(OUT/'affinity_loco_protocol.json')
    snapshot = ROOT/'experiments/v3/affinity_curated.json'
    rows = [r for r in read(snapshot) if r['document_ids']]
    source = read(OUT/'affinity_robustness.json')['provenance']
    text = snapshot.read_text(encoding='utf8')
    assert source['snapshot_original_sha256'] in [hashlib.sha256(t.encode()).hexdigest() for t in (text, text.replace('\n', '\r\n'))]
    stamp = {'protocol': protocol, 'snapshot_original_sha256': source['snapshot_original_sha256'], 'source_lf_sha256': {
        p: hashlib.sha256((ROOT/p).read_text(encoding='utf8').encode()).hexdigest()
        for p in ('scripts/affinity_loco_v4.py', 'scripts/affinity_v4.py', 'src/dmease/affinity_research.py')}}
    x = fingerprints([r['smiles'] for r in rows], radius=2, bits=2048); y = np.array([r['pK'] for r in rows])
    report = {'provenance': stamp, 'groupings': {}}
    for method in protocol['groupings']:
        groups = grouping(rows, method == 'document_and_scaffold'); strata = {}
        for target, endpoint in sorted({(r['target'], r['endpoint']) for r in rows}):
            indices = [i for i, r in enumerate(rows) if (r['target'], r['endpoint']) == (target, endpoint)]
            folds = []; predictions = []
            for group in sorted({groups[i] for i in indices}):
                train = [i for i in indices if groups[i] != group]; test = [i for i in indices if groups[i] == group]
                fold = {'component': group, 'train': len(train), 'test': len(test)}
                if len(train) < 40:
                    folds.append({**fold, 'status': 'insufficient_training'}); continue
                model = Ridge(alpha=10, solver='lsqr', tol=1e-6).fit(x[train], y[train])
                p = model.predict(x[test]); mean = float(y[train].mean())
                folds.append({**fold, 'status': 'evaluated', 'baseline_MAE': float(np.abs(y[test]-mean).mean()),
                              'ridge_MAE': float(np.abs(y[test]-p).mean())})
                predictions.extend({'row_index': i, 'component': group, 'pK': float(y[i]), 'mean': mean, 'ridge': float(p[j])} for j, i in enumerate(test))
            evaluated = [f for f in folds if f['status'] == 'evaluated']
            result = {'rows': len(indices), 'evaluated_rows': len(predictions), 'coverage': len(predictions)/len(indices),
                      'total_components': len(folds), 'evaluated_components': len(evaluated), 'folds': folds, 'predictions': predictions}
            if len(evaluated) < 2 or len(predictions) < 15:
                result['status'] = 'insufficient_evaluation'
            else:
                gold = [p['pK'] for p in predictions]; ridge = [p['ridge'] for p in predictions]; mean = [p['mean'] for p in predictions]
                result.update(status='evaluated', pooled={'training_mean': regression_metrics(gold, mean), 'ridge_alpha10': regression_metrics(gold, ridge)},
                              component_balanced_MAE={'training_mean': float(np.mean([f['baseline_MAE'] for f in evaluated])),
                                                      'ridge_alpha10': float(np.mean([f['ridge_MAE'] for f in evaluated]))},
                              comparison=scaffold_bootstrap(gold, ridge, mean, [p['component'] for p in predictions]))
                result['comparison']['components'] = result['comparison'].pop('scaffolds')
                result['comparison']['interpretation'] = 'Descriptive paired connected-component bootstrap interval; overlapping fold training sets and prior hyperparameter selection limit inference.'
            strata[target+'_'+endpoint] = result
        report['groupings'][method] = strata
    report['completed_at'] = datetime.now(timezone.utc).isoformat(); save(OUT/'affinity_loco.json', report)
    print(json.dumps({method: {name: {k: v for k, v in r.items() if k in ('status', 'evaluated_rows', 'coverage', 'evaluated_components', 'pooled')}
                              for name, r in strata.items()} for method, strata in report['groupings'].items()}), flush=True)


if __name__ == '__main__':
    main()

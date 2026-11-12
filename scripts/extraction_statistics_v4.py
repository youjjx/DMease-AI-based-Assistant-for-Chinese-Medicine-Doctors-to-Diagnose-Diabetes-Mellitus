"""Paired case-level uncertainty for the validation-selected extraction method."""
import hashlib
import json
from pathlib import Path

import numpy as np

from dmease.experiment_metrics import holm_adjust

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'


def f1(counts):
    denominator = 2*counts[..., 0]+counts[..., 1]+counts[..., 2]
    return np.divide(2*counts[..., 0], denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)


def compare(left, right, seed=20261112, repeats=10000):
    a = {r['row_hash']: r['surface'] for r in left}; b = {r['row_hash']: r['surface'] for r in right}
    assert a.keys() == b.keys() and len(a) == len(left) == len(right)
    ids = sorted(a)
    x = np.array([[a[key][k] for k in ('tp', 'fp', 'fn')] for key in ids])
    y = np.array([[b[key][k] for k in ('tp', 'fp', 'fn')] for key in ids])
    observed = float(f1(x.sum(0))-f1(y.sum(0)))
    rng = np.random.default_rng(seed); indices = rng.integers(len(ids), size=(repeats, len(ids)))
    bootstrap = f1(x[indices].sum(1))-f1(y[indices].sum(1))
    swapped = rng.integers(2, size=(repeats*2, len(ids), 1)).astype(bool)
    permuted = f1(np.where(swapped, x, y).sum(1))-f1(np.where(swapped, y, x).sum(1))
    return {'cases': len(ids), 'micro_f1_difference': observed, 'case_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
            'paired_randomization_p': float((1+np.sum(np.abs(permuted) >= abs(observed)-1e-12))/(len(permuted)+1)),
            'seed': seed, 'bootstrap_repeats': repeats, 'permutation_repeats': repeats*2}


def main():
    read = lambda name: json.loads((OUT/f'{name}.json').read_text(encoding='utf8'))
    data = read('extraction_test'); selection = read('extraction_selection'); chosen = selection['selected']
    comparisons = {method: compare(data['methods'][chosen], data['methods'][method]) for method in ('symmap_literal', 'copy_clauses', 'v3_strict')}
    for method, p in zip(comparisons, holm_adjust([r['paired_randomization_p'] for r in comparisons.values()])):
        comparisons[method]['holm_p'] = p
    result = {'selected_on_validation': chosen, 'selection_frozen_at': selection['frozen_before_test_at'],
              'scope': 'Post-evaluation descriptive paired case bootstrap and prediction-swap randomization; fixed three comparisons, Holm correction. No new model selection.',
              'source_lf_sha256': {'scripts/extraction_statistics_v4.py': hashlib.sha256(Path(__file__).read_text(encoding='utf8').encode()).hexdigest()},
              'comparisons': comparisons}
    (OUT/'extraction_statistics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8', newline='\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

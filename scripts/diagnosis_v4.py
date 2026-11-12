"""Evaluate already frozen diagnosis checkpoints against independent expert labels."""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

import joblib
import numpy as np

from dmease.experiment_metrics import paired_statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/v4'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return re.sub(r'\s+', '', text).removesuffix('证')


def digest(text):
    return hashlib.sha256(re.sub(r'\s+', '', text).encode()).hexdigest()


def choice_labels(options, letters):
    mapping = {m.group(1): m.group(2).strip() for m in re.finditer(r'(?:^|[;；])\s*([A-Z])\s*[:：]\s*([^;；]+)', options)}
    keys = {x.strip() for x in re.split('[;；]', letters) if x.strip()}
    if not keys or not keys <= mapping.keys():
        raise ValueError('Expert choices are not aligned with syndrome options')
    return {norm(mapping[k]) for k in keys}


def main():
    protocol = read(OUT/'diagnosis_protocol.json')
    baseline_path = ROOT/'outputs/tcm_sd_classifier.joblib'
    weighted_path = ROOT/'outputs/research_v3/imbalance/selected.joblib'
    original = read(ROOT/'experiments/v2/supervised_syndrome.json')
    selection = read(ROOT/'experiments/v3/imbalance_selection.json')
    assert sha(baseline_path) == original['selection']['checkpoint_sha256']
    assert sha(weighted_path) == selection['selected_checkpoint_sha256']
    raw = ROOT/'data/raw/public/tcmeval_v4'
    source_manifest = read(OUT/'extraction_sources.json')
    for name, meta in source_manifest['files'].items():
        assert sha(raw/Path(name).name) == meta['sha256']
    records = read(raw/'Test_TCM_Data_v1.json')
    answers = {parts[0].strip(): parts for line in (raw/'Test_data_result.txt').read_text(encoding='utf-8-sig').splitlines()
               if line.strip() and (parts := line.split('@', 4))}
    assert len(records) == len(answers) == 50
    texts = [r['Clinical Data'][:4096] for r in records]
    gold = [choice_labels(r['Options of TCM Syndrome'], answers[r['Medical Record ID'].strip()][3]) for r in records]
    base = joblib.load(baseline_path)
    seen = set()
    for split in ('train', 'dev'):
        path = ROOT/f'data/raw/public/tcm_sd/{split}.json'
        for line in path.read_text(encoding='utf8').splitlines():
            r = json.loads(line)
            seen.add(digest('\n'.join(r.get(k, '') for k in base['protocol']['input_fields'])[:4096]))
    matrix = base['vectorizer'].transform(texts)
    models = {'frozen_unweighted_v2': base['classifier'], 'frozen_weighted_v3': joblib.load(weighted_path)}
    results = {}; paired = {}
    for name, model in models.items():
        labels = [norm(x) for x in model.classes_]
        ranking = np.argsort(-model.decision_function(matrix), axis=1, kind='stable')[:, :3]
        known = set(labels)
        rows = []
        for text, g, rank in zip(texts, gold, ranking):
            pred = [labels[i] for i in rank]
            rows.append({'row_hash': digest(text), 'seen_in_tcm_sd_training_or_dev': digest(text) in seen,
                         'known_gold': bool(g & known), 'gold_labels': len(g), 'known_gold_labels': len(g & known),
                         'top1': int(pred[0] in g), 'top3': int(bool(set(pred) & g))})
        groups = {}
        for group, subset in (('all', rows), ('unseen_text', [r for r in rows if not r['seen_in_tcm_sd_training_or_dev']])):
            eligible = [r for r in subset if r['known_gold']]
            groups[group] = {'records': len(subset), 'top1': float(np.mean([r['top1'] for r in subset])) if subset else None,
                             'top3': float(np.mean([r['top3'] for r in subset])) if subset else None,
                             'known_label_ceiling': len(eligible)/len(subset) if subset else None,
                             'conditional_top1': float(np.mean([r['top1'] for r in eligible])) if eligible else None}
        results[name] = {'groups': groups, 'rows': rows}
        paired[name] = [{'query_id': r['row_hash'], 'objective': r['top1']} for r in rows]
    report = {'protocol': protocol, 'source_lf_sha256': {str(Path(__file__).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(Path(__file__).read_text(encoding='utf8').encode()).hexdigest()},
              'checkpoint_sha256': {'baseline': sha(baseline_path), 'weighted': sha(weighted_path)},
              'sources_sha256': sha(OUT/'extraction_sources.json'), 'results': results,
              'paired_top1': paired_statistics(paired['frozen_weighted_v3'], paired['frozen_unweighted_v2']),
              'completed_at': datetime.now(timezone.utc).isoformat()}
    (OUT/'diagnosis_test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8', newline='\n')
    print(json.dumps({k: v['groups'] for k, v in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Post-training action-regret diagnosis on training and validation only; no refitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dmease.evidence_policy import EvidenceGraph
from dmease.planning_v4 import PlanningEnvironment, PlanningPolicy, training_states

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'
PRIVATE = ROOT/'outputs/research_v4/planning'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def metrics(q, mask, prediction, greedy):
    row = torch.arange(len(q)); best = q.masked_fill(~mask, -1e9).max(-1).values
    error = best-q[row, prediction]; prior_error = best-q[row, greedy]
    prior_good = prior_error <= 1e-6; chosen_good = error <= 1e-6
    return {'states': len(q), 'mean_best_continuation_regret': float(error.mean()),
            'optimal_first_action_rate': float(chosen_good.float().mean()),
            'greedy_optimal_first_actions': int(prior_good.sum()),
            'damaged_greedy_optimal_actions': int((prior_good & ~chosen_good).sum()),
            'greedy_suboptimal_first_actions': int((~prior_good).sum()),
            'rescued_greedy_suboptimal_actions': int((~prior_good & chosen_good).sum()),
            'mean_q_change_vs_greedy': float((q[row, prediction]-q[row, greedy]).mean())}


def main():
    torch.set_num_threads(1)
    protocol = read(OUT/'planning_protocol.json'); queries = read(OUT/'planning_queries.json')
    teacher = read(OUT/'planning_teacher.json')
    assert hashlib.sha256((PRIVATE/'teacher.pt').read_bytes()).hexdigest() == teacher['checkpoint_sha256']
    data = {'train': torch.load(PRIVATE/'teacher.pt', weights_only=True, map_location='cpu')['tensors']}
    graph = EvidenceGraph(ROOT/'data/processed/evidence_bundle.json'); assert graph.fingerprint == protocol['evidence_fingerprint']
    collected = [[], [], [], [], []]
    for task in queries['splits']['validation']:
        rows, _ = training_states(PlanningEnvironment(graph, task), seed=int(task['id'][:8], 16))
        for row in rows:
            for i, value in enumerate(row):
                collected[i].append(value)
    data['validation'] = [torch.stack(collected[i]) for i in range(4)]+[torch.tensor(collected[4])]
    output = {'scope': 'Post-training diagnostic, train and validation states only. No new fitting or checkpoint selection. Action regret assumes optimal continuation, not actual full-policy rollout regret.',
              'source_lf_sha256': {'scripts/planning_diagnostics_v4.py': hashlib.sha256(Path(__file__).read_text(encoding='utf8').encode()).hexdigest()},
              'planning_provenance': queries['provenance'], 'results': {}}
    for split, (obs, features, mask, q, _) in data.items():
        greedy = (features[:, :, 0]+features[:, :, 8]*1e-7).masked_fill(~mask, -1e9).argmax(-1)
        results = {'greedy': metrics(q, mask, greedy, greedy)}
        for variant in protocol['variants']:
            for seed in protocol['seeds']:
                meta = read(OUT/f'planning_training/{variant}_{seed}.json')
                policy = PlanningPolicy(obs.shape[1], features.shape[2], variant)
                for mode in ('warm', 'selected'):
                    path = PRIVATE/f'{variant}_{seed}{"_warm" if mode == "warm" else ""}.pt'
                    key = 'warm_checkpoint_sha256' if mode == 'warm' else 'checkpoint_sha256'
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta[key]
                    policy.load_state_dict(torch.load(path, weights_only=True, map_location='cpu')['state_dict']); policy.eval()
                    prediction = []
                    with torch.no_grad():
                        for indices in torch.arange(len(obs)).split(128):
                            scores = policy.actor(obs[indices], features[indices]).masked_fill(~mask[indices], -1e9)
                            prediction.append(scores.argmax(-1))
                    results[f'{variant}_{seed}_{mode}'] = metrics(q, mask, torch.cat(prediction), greedy)
        output['results'][split] = results
    (OUT/'planning_diagnostics.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf8', newline='\n')
    print(json.dumps(output['results']['validation'], indent=2), flush=True)


if __name__ == '__main__':
    main()

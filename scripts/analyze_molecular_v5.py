"""Describe deployed-model and chemistry contributions without test-time reselection."""
from collections import Counter
import numpy as np

from chemistry_v5 import ROOT,OUT,read,save
from dmease.experiment_metrics import paired_statistics


def main():
    methods=read(OUT/'planning_results.json')['methods'];selection=read(OUT/'planning_selection.json')
    key=selection['final_ppo_run'];meta=read(OUT/'planning_training'/(key+'.json'))
    network=next(r['rows'] for r in methods[meta['variant']+'_final'] if r['run']==key)
    greedy={r['query_id']:r for r in methods['greedy'][0]['rows']}
    guarded=[{**(r if r['objective']>greedy[r['query_id']]['objective']+1e-7 else greedy[r['query_id']]),
              'used_network':r['objective']>greedy[r['query_id']]['objective']+1e-7} for r in network]
    original=methods['no_molecular_greedy'][0]['rows'];measured=methods['measured_only_greedy'][0]['rows']
    summary={}
    for name,rows in [('molecular_greedy',list(greedy.values())),('final_ppo',network),('final_ppo_guarded',guarded),
                      ('without_molecular',original),('measured_only',measured)]:
        summary[name]={'objective':float(np.mean([r['objective'] for r in rows])),
                       'molecular_index':float(np.mean([r['molecular_index'] for r in rows])),
                       'symptom_coverage':float(np.mean([r['symptom_coverage'] for r in rows])),
                       'mean_herbs':float(np.mean([len(r['selected_herbs']) for r in rows])),
                       'herb_selection_counts':dict(Counter(h for r in rows for h in r['selected_herbs'])),
                       'hard_violations':sum(r['hard_violations'] for r in rows)}
    bases=sorted({r['base_id'] for r in network})
    def groups(rows):return [{'query_id':b,'objective':float(np.mean([r['objective'] for r in rows if r['base_id']==b]))} for b in bases]
    changed=sum(set(r['selected_herbs'])!=set(greedy[r['query_id']]['selected_herbs']) for r in original)
    output={'deployment_selection':selection,'summary':summary,'chemistry_changes_selection':{'queries':len(original),'changed':changed},
            'final_ppo_vs_greedy_descriptive':paired_statistics(groups(network),groups(list(greedy.values()))),
            'guard_used_network':sum(r['used_network'] for r in guarded),'guard_rows':guarded,
            'scope':'Final model/run chosen using validation only. Guard explicitly computes two trajectories; any gain from the guard is not attributed to raw PPO. All ablations rescored under combined objective.'}
    save(OUT/'integration_analysis.json',output)
    print({k:v for k,v in output.items() if k in ('chemistry_changes_selection','guard_used_network')},flush=True)


if __name__=='__main__':main()

"""Train/evaluate KAN/PPO with actual constituent affinity evidence in its reward."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import time

import numpy as np
import torch

from chemistry_v5 import ROOT, OUT, read, save, sha
from dmease.evidence_policy import EvidenceGraph, canonical_hash
from dmease.molecular_planning import MolecularEvidence, MolecularEnvironment
from dmease.planning_v4 import PlanningPolicy, PlannerPPO, rollout, training_states
from dmease.experiment_metrics import paired_statistics, holm_adjust

PRIVATE=ROOT/'outputs/research_v5/planning'
MODELS=ROOT/'models/planning_v5'
SOURCES=['scripts/planning_v5.py','src/dmease/molecular_planning.py','src/dmease/planning_v4.py',
         'src/dmease/evidence_policy.py','src/dmease/research_policy.py','src/dmease/experiment_metrics.py']


def provenance(graph,chem):
    return {'protocol_sha256':sha(OUT/'protocol.json'),'evidence':graph.fingerprint,'molecular_evidence':chem.fingerprint,
            'source_lf_sha256':{p:hashlib.sha256((ROOT/p).read_text(encoding='utf8').encode()).hexdigest() for p in SOURCES},
            'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu','threads':1}


def prepare(graph,chem,stamp):
    used=set()
    for filename in ('experiments/queries.json','experiments/v2/queries.json','experiments/v4/planning_queries.json'):
        for rows in read(ROOT/filename)['splits'].values():
            used.update(canonical_hash(sorted(r['symptoms'])) for r in rows)
    bases=[]
    for count in (3,4,5):
        for names in itertools.combinations(sorted(graph.snames),count):
            identity=canonical_hash(sorted(names))
            if identity not in used:bases.append((identity,names))
    bases.sort(key=lambda x:hashlib.sha256(('v5-molecular:'+x[0]).encode()).hexdigest())
    splits={};start=0
    for label,count in (('train',160),('validation',40),('test',80)):
        tasks=[]
        for base,names in bases[start:start+count]:
            rng=np.random.default_rng(int(base[:8],16));herbs=[h['name'] for h in graph.herbs]
            forbidden=rng.choice(herbs,2,replace=False).tolist()
            paired=rng.choice([h for h in herbs if h not in forbidden],4,replace=False).tolist()
            for profile in ('standard','requested_constraints'):
                task={'base_id':base,'symptoms':dict.fromkeys(names,1.),'profile':profile,
                      'budget':8 if profile=='standard' else 4,'forbidden':[] if profile=='standard' else forbidden,
                      'forbidden_pairs':[] if profile=='standard' else [paired[:2],paired[2:]],
                      'molecular_targets':list(chem.data['targets']),
                      'scope':'Engineering evidence-retrieval request, not a patient or pharmacological contraindication'}
                task['id']=canonical_hash(task);tasks.append(task)
        splits[label]=tasks;start+=count
    save(OUT/'planning_queries.json',{'provenance':stamp,'splits':splits,'excluded_previous_bases':len(used)})
    tensors=[[],[],[],[],[]];audit=[]
    for i,task in enumerate(splits['train']):
        env=MolecularEnvironment(graph,task,chem)
        rows,meta=training_states(env,seed=int(task['id'][:8],16))
        for row in rows:
            for j,value in enumerate(row):tensors[j].append(value)
        audit.append({'query_id':task['id'],**meta})
        if (i+1)%20==0:print('Exact molecular teacher',i+1,'/',len(splits['train']),flush=True)
    data=[torch.stack(tensors[i]) for i in range(4)]+[torch.tensor(tensors[4])]
    PRIVATE.mkdir(parents=True,exist_ok=True);torch.save({'tensors':data,'provenance':stamp},PRIVATE/'teacher.pt')
    save(OUT/'planning_teacher.json',{'provenance':stamp,'states':len(data[0]),'rows':audit,'checkpoint_sha256':sha(PRIVATE/'teacher.pt')})


def train(graph,chem,stamp,variant,seed,config):
    tasks=read(OUT/'planning_queries.json');assert tasks['provenance']==stamp
    assert sha(PRIVATE/'teacher.pt')==read(OUT/'planning_teacher.json')['checkpoint_sha256']
    data=torch.load(PRIVATE/'teacher.pt',weights_only=True);assert data['provenance']==stamp
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    envs=[MolecularEnvironment(graph,t,chem) for t in tasks['splits']['train']]
    validation=[MolecularEnvironment(graph,t,chem) for t in tasks['splits']['validation']]
    policy=PlanningPolicy(envs[0].observation_dim,envs[0].feature_dim,variant,config['temperature'])
    history=[];best=-np.inf;chosen=None;best_state=None
    def observe(iteration):
        nonlocal best,chosen,best_state
        policy.eval();score=float(np.mean([rollout(e,policy)[1] for e in validation]))
        history.append({'iteration':iteration,'validation_objective':score})
        if score>best+1e-8:
            best=score;chosen=iteration;best_state={k:v.detach().clone() for k,v in policy.state_dict().items()}
        print(variant,seed,'iteration',iteration,'validation',round(score,7),flush=True)
    observe(-1);obs,feat,mask,q,values=data['tensors']
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=3e-4)
    for _ in range(config['warmup_epochs']):
        policy.train()
        for index in torch.randperm(len(obs)).split(64):
            p=policy.actor(obs[index],feat[index]).masked_fill(~mask[index],-1e9)
            target=q[index].masked_fill(~mask[index],-1e9)
            teacher=torch.softmax(target/config['temperature'],dim=-1)
            loss=-(teacher*torch.log_softmax(p/config['temperature'],dim=-1)).sum(-1).mean()
            pc=p-p.max(-1,keepdim=True).values;tc=target-target.max(-1,keepdim=True).values
            loss=loss+.5*((pc-tc).masked_fill(~mask[index],0)/.1).square().sum(-1).div(mask[index].sum(-1)).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
    critic=torch.optim.Adam(policy.critic.parameters(),lr=3e-4)
    for _ in range(config['critic_epochs']):
        for index in torch.randperm(len(obs)).split(128):
            loss=(policy.critic(obs[index]).squeeze(-1)-values[index]).square().mean()
            critic.zero_grad();loss.backward();critic.step()
    MODELS.mkdir(parents=True,exist_ok=True);key=f'{variant}_{seed}'
    def checkpoint(suffix,state,iteration):
        path=MODELS/(key+suffix+'.pt')
        torch.save({'state_dict':state,'provenance':stamp,'iteration':iteration},path)
        return sha(path)
    observe(0);warm_sha=checkpoint('_warm',policy.state_dict(),0)
    trainer=PlannerPPO(policy,actor_lr=1e-4,critic_lr=3e-4,target_kl=.02);diagnostics=[]
    for iteration in range(1,config['ppo_iterations']+1):
        policy.train();diag=trainer.update(trainer.collect(envs,rng,config['episodes']),.003*(1-iteration/config['ppo_iterations']))
        diagnostics.append({'iteration':iteration,**diag})
        if iteration%config['validation_interval']==0:observe(iteration)
    final_sha=checkpoint('_final',policy.state_dict(),config['ppo_iterations'])
    selected_sha=checkpoint('',best_state,chosen)
    save(OUT/'planning_training'/(key+'.json'),{'provenance':stamp,'variant':variant,'seed':seed,
        'selected_iteration':chosen,'validation_objective':best,'history':history,'ppo_diagnostics':diagnostics,
        'checkpoint_sha256':selected_sha,'warm_checkpoint_sha256':warm_sha,'final_checkpoint_sha256':final_sha,
        'observation_dim':envs[0].observation_dim,'feature_dim':envs[0].feature_dim,
        'actor_parameters':sum(p.numel() for p in policy.actor.parameters()),
        'frozen_before_test_at':datetime.now(timezone.utc).isoformat()})


def result(env,selected,seconds):
    names=[env.graph.herbs[i]['name'] for i in range(env.h) if selected&(1<<i)]
    violations=len(set(names)&set(env.task['forbidden']))+sum(a in names and b in names for a,b in env.task['forbidden_pairs'])+int(len(names)>env.budget)
    score=env.score(selected)
    return {'query_id':env.task['id'],'base_id':env.task['base_id'],'profile':env.task['profile'],'selected_herbs':names,
            'objective':score[0],'symptom_coverage':score[1],'target_coverage':score[2],
            'molecular_index':score[3],'seconds':seconds,'hard_violations':violations}


def evaluate(graph,chem,stamp,config):
    training={f'{v}_{s}':read(OUT/'planning_training'/f'{v}_{s}.json') for v in config['models'] for s in config['seeds']}
    assert all(r['provenance']==stamp for r in training.values())
    eligible={k:r for k,r in training.items() if r['variant']!='residual_mlp'}
    selected=max(eligible,key=lambda k:eligible[k]['validation_objective'])
    final_selected=max(eligible,key=lambda k:eligible[k]['history'][-1]['validation_objective'])
    selection_path=OUT/'planning_selection.json'
    selection={'provenance':stamp,'selected_kan_run':selected,'validation_objective':eligible[selected]['validation_objective'],
               'selected_iteration':eligible[selected]['selected_iteration'],
               'final_ppo_run':final_selected,'final_ppo_validation_objective':eligible[final_selected]['history'][-1]['validation_objective'],
               'training_metadata_sha256':{k:sha(OUT/'planning_training'/(k+'.json')) for k in training}}
    if selection_path.exists():
        previous=read(selection_path);assert {k:previous[k] for k in selection}==selection
    else:save(selection_path,{**selection,'frozen_before_test_at':datetime.now(timezone.utc).isoformat()})
    # Test labels/objectives are first used below, after all model/run selections.
    tasks=read(OUT/'planning_queries.json')['splits']['test'];envs=[MolecularEnvironment(graph,t,chem) for t in tasks]
    methods={}
    for method in ('greedy','lookahead2','exact','no_molecular_greedy','measured_only_greedy'):
        rows=[]
        for env in envs:
            started=time.perf_counter()
            if method=='exact':
                masks,scores=env.exact_table();chosen=int(masks[scores.argmax()])
            elif method in ('no_molecular_greedy','measured_only_greedy'):
                ablation=MolecularEnvironment(graph,env.task,chem,weight=0 if method=='no_molecular_greedy' else .35,
                                              mode='combined' if method=='no_molecular_greedy' else 'measured_only')
                chosen,_=rollout(ablation)
            else:chosen,_=rollout(env,method=method)
            rows.append(result(env,chosen,time.perf_counter()-started))
        methods[method]=[{'seed':None,'rows':rows}]
        print('Evaluated',method,flush=True)
    for key,meta in training.items():
        policy=PlanningPolicy(meta['observation_dim'],meta['feature_dim'],meta['variant'],config['temperature'])
        for suffix,label in (('',''),('_warm','_warm'),('_final','_final')):
            path=MODELS/(key+suffix+'.pt');assert sha(path)==meta[{'':'checkpoint_sha256','_warm':'warm_checkpoint_sha256','_final':'final_checkpoint_sha256'}[suffix]]
            payload=torch.load(path,weights_only=True,map_location='cpu');assert payload['provenance']==stamp
            policy.load_state_dict(payload['state_dict']);policy.eval();rows=[]
            for env in envs:
                started=time.perf_counter();chosen,_=rollout(env,policy)
                rows.append(result(env,chosen,time.perf_counter()-started))
            methods.setdefault(meta['variant']+label,[]).append({'seed':meta['seed'],'rows':rows,'run':key})
        print('Evaluated',key,flush=True)
    save(OUT/'planning_results.json',{'provenance':stamp,'methods':methods,'completed_at':datetime.now(timezone.utc).isoformat()})
    summarize(methods,selection)


def summarize(methods,selection):
    exact={r['query_id']:r['objective'] for r in methods['exact'][0]['rows']}
    greedy={r['query_id']:r for r in methods['greedy'][0]['rows']}
    summary={};statistics={}
    for method,runs in methods.items():
        flat=[r for run in runs for r in run['rows']];row={}
        for profile in ('all','standard','requested_constraints'):
            chosen=[r for r in flat if profile=='all' or r['profile']==profile]
            row[profile]={'objective':float(np.mean([r['objective'] for r in chosen])),
                'molecular_index':float(np.mean([r['molecular_index'] for r in chosen])),
                'mean_gap_to_optimal':float(np.mean([exact[r['query_id']]-r['objective'] for r in chosen])),
                'hard_violations':sum(r['hard_violations'] for r in chosen),
                'different_from_greedy':sum(set(r['selected_herbs'])!=set(greedy[r['query_id']]['selected_herbs']) for r in chosen),
                'median_ms':float(np.median([r['seconds'] for r in chosen])*1000)}
        summary[method]=row
        if method in ('residual_kan','residual_mlp','kan_without_prior'):
            bases=sorted({r['base_id'] for r in flat})
            network=[np.mean([r['objective'] for r in flat if r['base_id']==b]) for b in bases]
            baseline=[np.mean([r['objective'] for r in greedy.values() if r['base_id']==b]) for b in bases]
            statistics[method]=paired_statistics([{'query_id':b,'objective':float(x)} for b,x in zip(bases,network)],
                                                [{'query_id':b,'objective':float(x)} for b,x in zip(bases,baseline)],seed=42)
    for name,adjusted in zip(statistics,holm_adjust([r['sign_flip_p'] for r in statistics.values()])):
        statistics[name]['holm_p_three_network_comparisons']=adjusted
    save(OUT/'planning_summary.json',{'selection':selection,'summary':summary,'paired_by_symptom_base':statistics,
         'comparison_scope':'All ablation candidate sets rescored under the SAME combined objective; timings include their own proposal. No clinical endpoints.',
         'selected_iterations':{v:[read(OUT/'planning_training'/f'{v}_{s}.json')['selected_iteration'] for s in (42,43,44)]
                                for v in ('residual_kan','residual_mlp','kan_without_prior')}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate']);parser.add_argument('--variant');parser.add_argument('--seed',type=int)
    args=parser.parse_args();torch.set_num_threads(1)
    graph=EvidenceGraph(ROOT/'data/processed/evidence_bundle.json');chem=MolecularEvidence(OUT/'molecular_bundle.json',graph)
    stamp=provenance(graph,chem);config=read(OUT/'protocol.json')['planning']
    if args.stage=='prepare':prepare(graph,chem,stamp)
    elif args.stage=='train':
        if args.variant not in config['models'] or args.seed not in config['seeds']:raise ValueError('Undeclared training run')
        train(graph,chem,stamp,args.variant,args.seed,config)
    else:evaluate(graph,chem,stamp,config)


if __name__=='__main__':main()

"""Verified molecular KAN/PPO inference shared by patient and research workflows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .molecular_planning import MolecularEvidence, MolecularEnvironment
from .planning_v4 import PlanningPolicy, greedy_action
from .types import HerbRecommendation


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def load_policy(root, graph, chemistry, final=True):
    root=Path(root);selection=read(root/'experiments/v5/planning_selection.json')
    key=selection['final_ppo_run' if final else 'selected_kan_run']
    meta=read(root/f'experiments/v5/planning_training/{key}.json');stamp=meta['provenance']
    if stamp['evidence']!=graph.fingerprint or stamp['molecular_evidence']!=chemistry.fingerprint:
        raise ValueError('分子策略与当前证据版本不匹配，请重新训练。')
    if stamp!=selection['provenance']:
        raise ValueError('策略选择与训练来源不一致。')
    for name in ('src/dmease/molecular_planning.py','src/dmease/planning_v4.py','src/dmease/evidence_policy.py','src/dmease/research_policy.py'):
        if hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest()!=stamp['source_lf_sha256'][name]:
            raise ValueError('当前计算方法与模型训练版本不一致。')
    suffix='_final' if final else '';path=root/f'models/planning_v5/{key}{suffix}.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['final_checkpoint_sha256' if final else 'checkpoint_sha256']:
        raise ValueError('分子策略模型文件校验失败。')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['provenance']!=stamp:raise ValueError('模型内置来源记录不一致。')
    policy=PlanningPolicy(meta['observation_dim'],meta['feature_dim'],meta['variant'])
    policy.load_state_dict(payload['state_dict']);policy.eval()
    return policy,{'run':key,'iteration':payload['iteration'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'selection':'final PPO run chosen by validation' if final else 'checkpoint chosen by validation'}


def trajectory(env,policy=None):
    env.reset();steps=[];stop='达到候选数量或合法候选限制'
    while not env.done:
        probability=None;sensitivity={}
        if policy is not None:
            obs,features,mask=env.state();obs=obs.detach().requires_grad_(True)
            distribution,_=policy.distribution(obs[None],features[None],mask[None]);action=int(distribution.probs.argmax())
            probability=float(distribution.probs[0,action].detach())
            derivative=torch.autograd.grad(distribution.log_prob(torch.tensor([action])).sum(),obs)[0]
            sensitivity={s['name']:float(derivative[i]) for i,s in enumerate(env.graph.symptoms) if env.base.initial[i]>0}
        else:action=greedy_action(env)
        if action==env.stop:
            env.step(action);stop='策略选择停止';break
        before=env.score(env.selected);reward=env.step(action);after=env.score(env.selected)
        covered,_,_=env.base.coverage(env.vector(env.selected))
        steps.append({'step':len(steps)+1,'action':action,'herb':env.graph.herbs[action]['name'],
            'marginal_evidence_gain':reward,'molecular_index_before':before[3],'molecular_index_after':after[3],
            'policy_probability':probability,'local_symptom_gradient':sensitivity,
            'residual_evidence':{s['name']:float(env.base.initial[i]*(1-covered[i])) for i,s in enumerate(env.graph.symptoms) if env.base.initial[i]>0},
            'molecular_paths':[row for row in env.chemistry.trace(action,env.mode)
                               if env.requested[env.chemistry.channel_index[row['channel']]]]})
    return {**env.explain(),'steps':steps,'stop_reason':stop,
            'sensitivity_scope':'Partial derivative of log policy probability w.r.t. symptom observation coordinates, holding derived action features fixed; not a causal/clinical attribution'}


def recommend(graph, task, root, strategy='molecular_kan', weight=.35, mode='combined'):
    if strategy not in ('molecular_kan','molecular_greedy','molecular_guarded','molecular_selected'):
        raise ValueError('未知分子推荐策略。')
    if set(task.get('symptoms',{}))-set(graph.snames) or not graph.vector(task.get('symptoms',{})).any():
        raise ValueError('分子推荐需要至少一个当前词表中的阳性症状，请核对输入。')
    if set(task.get('forbidden',[]))-set(graph.hnames):
        raise ValueError('排除列表含当前候选范围以外的药材，请核对名称。')
    if any(len(pair)!=2 or len(set(pair))!=2 or not set(pair)<=set(graph.hnames) for pair in task.get('forbidden_pairs',[])):
        raise ValueError('配对排除必须填写两味不同的当前药材。')
    if isinstance(task.get('budget'),bool) or not isinstance(task.get('budget'),int):
        raise ValueError('候选数量必须为整数。')
    root=Path(root);chem=MolecularEvidence(root/'experiments/v5/molecular_bundle.json',graph)
    env=MolecularEnvironment(graph,task,chem,weight=weight,mode=mode)
    policy=None;metadata={}
    if strategy!='molecular_greedy':
        policy,metadata=load_policy(root,graph,chem,final=strategy!='molecular_selected')
    result=trajectory(env,policy);result.update(strategy=strategy,mode=strategy,model=metadata,
        evidence_mode=mode,task=task,used='greedy' if policy is None else 'network')
    if strategy=='molecular_guarded':
        baseline=trajectory(MolecularEnvironment(graph,task,chem,weight=weight,mode=mode))
        comparison={'network_objective':result['objective'],'greedy_objective':baseline['objective'],
                    'network_selected':result['selected'],'greedy_selected':baseline['selected']}
        if baseline['objective']>=result['objective']-1e-7:
            result={**baseline,'strategy':strategy,'mode':strategy,'evidence_mode':mode,'task':task,'model':metadata,'used':'greedy'}
        result['guard_comparison']=comparison
    result['out_of_training_configuration']=bool(weight!=.35 or mode!='combined' or
        set(task.get('molecular_targets',chem.data['targets']))!=set(chem.data['targets']) or task['budget'] not in (4,8))
    return result


def analyze(service,patient,selected_symptoms,top_herbs,root,strategy):
    # Reuse the established parser, patient validation, diagnosis and hard rules.
    result=service.analyze(patient,selected_symptoms,top_herbs,strategy='greedy')
    hard={h:[r for r in reasons if not r.startswith('与已选')] for h,reasons in result.excluded_herbs.items()}
    hard={h:reasons for h,reasons in hard.items() if reasons}
    task={'symptoms':{s.canonical:s.severity for s in result.parsed_symptoms if s.present},
          'forbidden':list(hard),'forbidden_pairs':patient.forbidden_pairs,'budget':min(top_herbs,service.evidence.action_dim)}
    molecular=recommend(service.evidence,task,root,strategy)
    result.recommendations=[]
    for step in molecular['steps']:
        paths=service.evidence.paths(task['symptoms'],step['action'])
        for row in step['molecular_paths']:
            kind='实测' if row['kind']=='measured' else '结构模型预测'
            paths.append(f"{step['herb']} → {row['tcmsp_id']} / PubChem CID {row['cid']} → {row['uniprot']} {row['endpoint']}；{kind} pK={row['pK']:.3f}；{row['pubchem_url']}")
        result.recommendations.append(HerbRecommendation(step['herb'],step['marginal_evidence_gain'],step['step'],[],[],paths,
            ['成分亲和度证据用于研究检索；缺少药材暴露浓度及作用方向，不能据此推断疗效或给药剂量。']))
    chosen=set(molecular['selected'])
    for a,b in patient.forbidden_pairs:
        if a in chosen and b not in chosen:hard.setdefault(b,[]).append(f'与已选{a}命中本次配伍排除')
        if b in chosen and a not in chosen:hard.setdefault(a,[]).append(f'与已选{b}命中本次配伍排除')
    result.excluded_herbs=hard
    result.metadata.update(ranking_mode=strategy,requested_strategy=strategy,trace=molecular,molecular_evidence=molecular,
                           model=molecular['model'],source_urls=sorted(set(result.metadata['source_urls'])|{r['pubchem_url'] for r in molecular['traces']}))
    result.safety_notes.append('分子索引区分实测、预测和缺失值；Ki与Kd分开计算，未把结合强度解释为治疗方向。')
    return result

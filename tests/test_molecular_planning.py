import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dmease.evidence_policy import EvidenceGraph, canonical_hash
from dmease.molecular_planning import MolecularEvidence, MolecularEnvironment
from dmease.planning_v4 import rollout, PlanningPolicy

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def molecular():
    path=ROOT/'data/processed/evidence_bundle.json'
    if path.exists():graph=EvidenceGraph(path)
    else:
        from dmease.molecular_feature_graph import PlanningFeatureGraph
        graph=PlanningFeatureGraph(ROOT/'experiments/v5/planning_features.json')
    return graph,MolecularEvidence(ROOT/'experiments/v5/molecular_bundle.json',graph)


def task(graph):
    return {'symptoms':dict.fromkeys(list(graph.snames)[:4],1.),'budget':4,'forbidden':['丹参'],
            'forbidden_pairs':[['黄芪','山茱萸']]}


def test_exact_rewards_and_constraints(molecular):
    graph,chem=molecular;env=MolecularEnvironment(graph,task(graph),chem)
    masks,scores=env.exact_table()
    rng=np.random.default_rng(42)
    for i in rng.choice(len(masks),min(40,len(masks)),replace=False):
        assert scores[i]==pytest.approx(env.score(int(masks[i]))[0],abs=2e-7)
    selected,score=rollout(env)
    assert not selected&(1<<graph.hnames['丹参'])
    assert not (selected&(1<<graph.hnames['黄芪']) and selected&(1<<graph.hnames['山茱萸']))
    assert selected.bit_count()<=4
    assert score<=scores.max()+2e-7


def test_affinity_changes_state_and_rewards(molecular):
    graph,chem=molecular;t=task(graph);t['forbidden']=[]
    env=MolecularEnvironment(graph,t,chem);zero=MolecularEnvironment(graph,t,chem,weight=0)
    selected=1<<graph.hnames['丹参']
    assert env.score(selected)[0]!=pytest.approx(zero.score(selected)[0])
    obs,features,mask=env.state()
    assert len(obs)==env.observation_dim and features.shape==(graph.action_dim+1,env.feature_dim)
    assert features[:,0].numpy()==pytest.approx([env.score(1<<i)[0]-env.score(0)[0] if mask[i] else 0 for i in range(graph.action_dim)]+[0],abs=1e-7)
    policy=PlanningPolicy(env.observation_dim,env.feature_dim,'kan_without_prior')
    distribution,_=policy.distribution(obs[None],features[None],mask[None])
    assert float(distribution.probs.sum())==pytest.approx(1.)


def test_missing_is_masked_and_unknown_target_disables_channel(molecular):
    graph,chem=molecular;t=task(graph);t['molecular_targets']=['INSR']
    env=MolecularEnvironment(graph,t,chem)
    assert env.effective_weight==0 and not env.available.any()
    assert all(x['index'] is None for x in env.explain()['channels'])
    with pytest.raises(ValueError):MolecularEnvironment(graph,{**t,'molecular_targets':['FAKE']},chem)
    with pytest.raises(ValueError):MolecularEnvironment(graph,t,chem,weight=float('nan'))


def test_duplicate_ingredient_does_not_inflate_and_measured_separate(molecular):
    graph,chem=molecular;before=chem.matrices()[0].copy()
    herb=next(iter(chem.data['herb_compounds']));chem.data['herb_compounds'][herb]*=2
    assert np.array_equal(before,chem.matrices()[0])
    _,_,winners=chem.matrices('measured_only')
    assert winners and all(v['affinity']['kind']=='measured' for v in winners.values())
    assert all(row['activity_ids'] for i in range(graph.action_dim) for row in chem.trace(i,'measured_only'))


def test_bundle_tamper_and_graph_mismatch(molecular,tmp_path):
    graph,chem=molecular;data=copy.deepcopy(chem.data);data['affinities'][0]['index']=.999
    path=tmp_path/'bad.json';path.write_text(json.dumps(data),encoding='utf8')
    with pytest.raises(ValueError):MolecularEvidence(path,graph)
    data.pop('fingerprint');data['graph_fingerprint']='wrong';data['fingerprint']=canonical_hash(data)
    path.write_text(json.dumps(data),encoding='utf8')
    with pytest.raises(ValueError):MolecularEvidence(path,graph)


def test_runtime_returns_real_paths_and_telescoping_rewards(molecular):
    from dmease.molecular_runtime import recommend
    graph,chem=molecular
    result=recommend(graph,task(graph),ROOT,'molecular_greedy')
    assert result['evidence_fingerprint']==chem.fingerprint
    assert sum(r['marginal_evidence_gain'] for r in result['steps'])==pytest.approx(result['objective'])
    assert result['used']=='greedy' and result['model']=={}
    for row in result['traces']:
        assert str(row['cid']) in chem.data['compounds']
        assert row['ranking_pK']<=row['pK']
        if row['kind']=='predicted':assert row['nearest_tanimoto']>=.5


def test_all_excluded_stops_without_unmasking(molecular):
    from dmease.molecular_runtime import recommend
    graph,_=molecular;t=task(graph);t['forbidden']=list(graph.hnames)
    result=recommend(graph,t,ROOT,'molecular_greedy')
    assert result['selected']==[] and result['steps']==[] and result['objective']==0


def test_parent_matching_preserves_stereochemistry():
    from dmease.affinity_research import standardized_structure
    from rdkit import Chem
    a,_=standardized_structure('F[C@H](Cl)Br');b,_=standardized_structure('F[C@@H](Cl)Br')
    assert a!=b
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(a))!=Chem.MolToInchiKey(Chem.MolFromSmiles(b))


def test_runtime_rejects_unknown_or_empty_requests(molecular):
    from dmease.molecular_runtime import recommend
    graph,_=molecular
    for patch in ({'symptoms':{}},{'forbidden':['不在词表内']},{'forbidden_pairs':[['黄芪','黄芪']]},{'budget':4.5}):
        with pytest.raises(ValueError):recommend(graph,{**task(graph),**patch},ROOT,'molecular_greedy')

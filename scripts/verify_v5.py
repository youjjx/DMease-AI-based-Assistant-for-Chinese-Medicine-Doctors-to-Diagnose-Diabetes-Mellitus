"""Offline validation of identity, affinity provenance, checkpoints and objectives."""
from collections import Counter
import hashlib
import itertools

import numpy as np
import torch
from rdkit import Chem

from chemistry_v5 import ROOT, OUT, read, save, sha, formula_elements
from dmease.evidence_policy import canonical_hash, EvidenceEnvironment
from dmease.molecular_feature_graph import PlanningFeatureGraph
from dmease.molecular_planning import MolecularEvidence, MolecularEnvironment
from dmease.affinity_research import fingerprints


def main():
    torch.set_num_threads(1)
    bundle=read(OUT/'molecular_bundle.json');graph=PlanningFeatureGraph(OUT/'planning_features.json')
    chem=MolecularEvidence(OUT/'molecular_bundle.json',graph)
    for name,expected in bundle['source_files'].items():assert sha(OUT/name)==expected,name
    assert sha(OUT/'protocol.json')==bundle['protocol_sha256']
    assert sha(OUT/'affinity_models.json')==bundle['affinity_models_sha256']
    identity=read(OUT/'identity_audit.json');source_links=read(OUT/'identifier_links.json')
    assert len(identity['accepted'])+len(identity['rejected'])==len(source_links['links'])
    assert identity['counts']['accepted_links']==bundle['audit']['resolved_herb_compound_links']
    for link in identity['accepted']:
        structure=bundle['compounds'][str(link['cid'])];mw=float(structure['MolecularWeight'])
        assert abs(float(link['tcmsp_molecular_weight'])-mw)<=max(1.,mw*.005)
        for formula in link['symmap_formulas']:
            elements=formula_elements(formula)
            if elements:assert elements==formula_elements(structure['MolecularFormula'])
    for s in bundle['compounds'].values():
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(s['SMILES']))==s['InChIKey']
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(s['parent_smiles']))==s['parent_inchikey']
    affinity=read(OUT/'affinity_evaluation.json');models=read(OUT/'affinity_models.json')['models']
    rows=affinity['filtered_rows'];query=fingerprints([s['parent_smiles'] for s in bundle['compounds'].values()])
    cids=list(map(int,bundle['compounds']));lookup={cid:i for i,cid in enumerate(cids)}
    similarities={}
    for name,model in models.items():
        train=fingerprints([rows[i]['smiles'] for i in model['training_row_indices']])
        intersection=query@train.T
        similarities[name]=(intersection/np.maximum(query.sum(1)[:,None]+train.sum(1)[None,:]-intersection,1)).max(1)
    prediction_error=0
    for r in bundle['affinities']:
        assert r['index']==float(np.clip((r['ranking_pK']-4)/5,0,1))
        structure=bundle['compounds'][str(r['cid'])]
        matches=[v for v in rows if v['target']==r['target'] and v['endpoint']==r['endpoint'] and v['smiles']==structure['parent_smiles']]
        if r['kind']=='measured':
            assert len(matches)==1 and matches[0]['pK']==r['pK'] and matches[0]['activity_ids']==r['activity_ids']
        else:
            assert not matches and r['nearest_tanimoto']>=.5
            assert abs(float(similarities[r['model']][lookup[r['cid']]])-r['nearest_tanimoto'])<1e-7
            model=models[r['model']]
            predicted=float(query[lookup[r['cid']]]@np.array(model['coefficients'])+model['intercept'])
            error=abs(predicted-r['pK']);prediction_error=max(prediction_error,error);assert error<2e-6
            assert abs(r['ranking_pK']-(r['pK']-model['absolute_error_q90']))<1e-8
    for name,expected in read(OUT/'build_stamp.json')['source_lf_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf8').encode()).hexdigest()==expected,name
    selection=read(OUT/'planning_selection.json');stamp=selection['provenance']
    assert stamp['molecular_evidence']==chem.fingerprint and stamp['protocol_sha256']==sha(OUT/'protocol.json')
    for name,expected in stamp['source_lf_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf8').encode()).hexdigest()==expected,name
    for key,expected in selection['training_metadata_sha256'].items():
        meta_path=OUT/'planning_training'/(key+'.json');assert sha(meta_path)==expected
        meta=read(meta_path);assert meta['provenance']==stamp
        for suffix,field in (('','checkpoint_sha256'),('_warm','warm_checkpoint_sha256'),('_final','final_checkpoint_sha256')):
            path=ROOT/'models/planning_v5'/(key+suffix+'.pt');assert sha(path)==meta[field]
            assert torch.load(path,weights_only=True,map_location='cpu')['provenance']==stamp
    queries=read(OUT/'planning_queries.json')['splits'];sets=[{r['base_id'] for r in rows} for rows in queries.values()]
    assert not any(a&b for a,b in itertools.combinations(sets,2))
    previous=set()
    for path in ('experiments/queries.json','experiments/v2/queries.json','experiments/v4/planning_queries.json'):
        for rows in read(ROOT/path)['splits'].values():previous.update(canonical_hash(sorted(r['symptoms'])) for r in rows)
    assert not set.union(*sets)&previous
    tasks={t['id']:t for t in queries['test']};results=read(OUT/'planning_results.json')['methods']
    matrix,known,_=chem.matrices();active=known.any(0)
    def reference(task,names):
        selected=np.zeros(graph.action_dim,dtype=np.float32);selected[[graph.hnames[h] for h in names]]=1
        base=EvidenceEnvironment(graph,max_steps=task['budget']);base.reset(task)
        old=base.coverage(selected)[2]['objective']
        coverage=(selected[:,None]*matrix).max(0)
        molecular=float(coverage[active].mean()) if active.any() else 0.
        return .65*old+.35*molecular-.35*.02*len(names)
    max_error=0.;count=0
    for runs in results.values():
        for run in runs:
            assert {r['query_id'] for r in run['rows']}==set(tasks)
            for row in run['rows']:
                task=tasks[row['query_id']];names=row['selected_herbs']
                assert len(names)==len(set(names))<=task['budget']
                assert not set(names)&set(task['forbidden'])
                assert not any(a in names and b in names for a,b in task['forbidden_pairs'])
                error=abs(reference(task,names)-row['objective']);max_error=max(max_error,error);assert error<2e-7
                count+=1
    exact={r['query_id']:r['objective'] for r in results['exact'][0]['rows']}
    assert all(row['objective']<=exact[row['query_id']]+2e-7 for runs in results.values() for run in runs for row in run['rows'])
    independent_exact=[]
    for task in [t for t in tasks.values() if t['profile']=='requested_constraints'][:8]:
        available=[h for h in graph.hnames if h not in task['forbidden']];best=0.;examined=0
        # Enumerate ALL allowed herbs, without the planner's usefulness pruning or vectorized table.
        for size in range(task['budget']+1):
            for names in itertools.combinations(available,size):
                if any(a in names and b in names for a,b in task['forbidden_pairs']):continue
                best=max(best,reference(task,names));examined+=1
        assert abs(best-exact[task['id']])<2e-7
        independent_exact.append({'query_id':task['id'],'feasible_subsets':examined,'objective':best})
    report={'molecular_fingerprint':chem.fingerprint,'identity_links_verified':len(identity['accepted']),
            'structures_verified':len(bundle['compounds']),'affinity_entries_verified':len(bundle['affinities']),
            'max_prediction_reconstruction_error':prediction_error,'planning_rows_rescored':count,'max_objective_error':max_error,
            'verified_checkpoints':len(selection['training_metadata_sha256'])*3,
            'independent_scalar_exact_checks':independent_exact,
            'test_base_groups':len(sets[-1]),'scope':'Offline feature-snapshot and artifact audit; molecular structure reconstruction is not independent chemical or clinical validation.'}
    save(OUT/'verification.json',report);print(report,flush=True)


if __name__=='__main__':main()

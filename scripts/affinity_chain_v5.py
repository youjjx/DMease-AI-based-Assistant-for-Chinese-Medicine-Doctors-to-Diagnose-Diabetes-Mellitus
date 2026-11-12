"""Fit audited structure models and join supported compound affinities to herbs."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib

import numpy as np
from rdkit import Chem
from sklearn.linear_model import Ridge

from chemistry_v5 import ROOT, OUT, read, save, sha
from affinity_v4 import grouping
from dmease.affinity_research import fingerprints, regression_metrics, scaffold_bootstrap
from dmease.evidence_policy import canonical_hash


def similarities(query, train):
    intersection = query @ train.T
    return intersection / np.maximum(query.sum(1)[:,None]+train.sum(1)[None,:]-intersection, 1)


def main():
    protocol=read(OUT/'protocol.json'); config=protocol['affinity_model']
    path=ROOT/'experiments/v3/affinity_curated.json'
    expected=read(ROOT/'experiments/v3/affinity_sources.json')['curated_sha256']
    text=path.read_text(encoding='utf8')
    assert expected in [hashlib.sha256(s.encode()).hexdigest() for s in (text,text.replace('\n','\r\n'))]
    original=read(path)
    rows=[r for r in original if r['document_ids'] and r['replicate_range']<=1]
    groups=grouping(rows,True)
    links=read(OUT/'identity_audit.json')['accepted'];valid_cids={r['cid'] for r in links}
    structures=[r for r in read(OUT/'structures.json')['records'] if r['cid'] in valid_cids]
    x=fingerprints([r['smiles'] for r in rows]); y=np.array([r['pK'] for r in rows])
    query=fingerprints([r['parent_smiles'] for r in structures])
    keys=[Chem.MolToInchiKey(Chem.MolFromSmiles(r['smiles'])) for r in rows]
    measurements=[]; predictions=[]; unsupported=[]; evaluations={}; models={}
    for target,endpoint in sorted({(r['target'],r['endpoint']) for r in rows}):
        name=target+'_'+endpoint; ids=[i for i,r in enumerate(rows) if (r['target'],r['endpoint'])==(target,endpoint)]
        folds=[]; held=[]
        for group in sorted({groups[i] for i in ids}):
            tr=[i for i in ids if groups[i]!=group]; te=[i for i in ids if groups[i]==group]
            fold={'component':group,'train_rows':len(tr),'test_rows':len(te)}
            if len(tr)<config['minimum_fit_rows']:
                folds.append({**fold,'status':'insufficient_training'});continue
            model=Ridge(alpha=10,solver='lsqr',tol=1e-6).fit(x[tr],y[tr])
            p=model.predict(x[te]); baseline=float(y[tr].mean())
            held.extend({'row':i,'component':group,'gold':float(y[i]),'prediction':float(p[j]),'baseline':baseline} for j,i in enumerate(te))
            folds.append({**fold,'status':'evaluated'})
        components=len({p['component'] for p in held})
        evaluation={'rows':len(ids),'folds':folds,'evaluated_rows':len(held),'evaluated_components':components,'predictions':held,'deployable':False}
        if len(held)>=2:
            gold=np.array([p['gold'] for p in held]); pred=np.array([p['prediction'] for p in held]); mean=np.array([p['baseline'] for p in held])
            evaluation.update(model=regression_metrics(gold,pred),baseline=regression_metrics(gold,mean),
                absolute_error_q90=float(np.quantile(np.abs(gold-pred),.9)),
                comparison=scaffold_bootstrap(gold,pred,mean,[p['component'] for p in held]))
            evaluation['deployable']=bool(len(ids)>=config['minimum_fit_rows'] and len(held)>=config['minimum_evaluated_rows'] and
                components>=config['minimum_evaluated_components'] and evaluation['model']['MAE_pK']<evaluation['baseline']['MAE_pK'])
        model=None
        if evaluation['deployable']:
            model=Ridge(alpha=10,solver='lsqr',tol=1e-6).fit(x[ids],y[ids]); values=model.predict(query)
            sim=similarities(query,x[ids]); nearest=sim.argmax(1)
            models[name]={'coefficients':model.coef_.tolist(),'intercept':float(model.intercept_),
                          'training_row_indices':ids,'absolute_error_q90':evaluation['absolute_error_q90'],
                          'minimum_tanimoto':config['minimum_tanimoto']}
        for j,structure in enumerate(structures):
            matches=[i for i in ids if keys[i]==structure['parent_inchikey'] and rows[i]['smiles']==structure['parent_smiles']]
            base={'cid':structure['cid'],'target':target,'uniprot':protocol['data']['targets'][target],'endpoint':endpoint}
            if len(matches)==1:
                r=rows[matches[0]]
                measurements.append({**base,'kind':'measured','pK':r['pK'],'ranking_pK':r['pK'],
                    'activity_ids':r['activity_ids'],'molecule_ids':r['molecule_ids'],'assay_ids':r['assay_ids'],
                    'document_ids':r['document_ids'],'replicate_count':r['replicate_count'],
                    'replicate_range':r['replicate_range'],'structure_match':'identical standardized isomeric parent AND full InChIKey'})
            elif len(matches)>1:
                raise ValueError('Ambiguous standardized measurement identity')
            elif model is not None and float(sim[j,nearest[j]])>=config['minimum_tanimoto']:
                pk=float(values[j]); band=evaluation['absolute_error_q90']; neighbor=rows[ids[int(nearest[j])]]
                predictions.append({**base,'kind':'predicted','pK':pk,'ranking_pK':pk-band,
                    'error_band':[pk-band,pk+band],'nearest_tanimoto':float(sim[j,nearest[j]]),
                    'nearest_molecule_ids':neighbor['molecule_ids'],'model':name,
                    'note':'Descriptive LOCO error band; lower bound used for evidence ranking, no calibrated probability'})
            else:
                unsupported.append({**base,'reason':'model_not_supported' if model is None else 'outside_applicability_domain',
                                    'nearest_tanimoto':None if model is None else float(sim[j,nearest[j]])})
        evaluations[name]=evaluation
        print(name,'rows',len(ids),'LOCO',len(held),'deploy',evaluation['deployable'],flush=True)
    save(OUT/'affinity_evaluation.json',{'protocol_sha256':sha(OUT/'protocol.json'),'source_snapshot_sha256':expected,
         'excluded_replicate_conflict_or_no_document':len(original)-len(rows),'filtered_rows':rows,'strata':evaluations})
    save(OUT/'affinity_models.json',{'features':config['features'],'source_snapshot_sha256':expected,'models':models})
    compounds={str(s['cid']):s for s in structures}; by_cid=defaultdict(list)
    for r in measurements+predictions:
        r['index']=float(np.clip((r['ranking_pK']-4)/5,0,1));by_cid[r['cid']].append(r)
    herb_names={r['herb_id']:r['herb'] for r in links}; herb_links=defaultdict(list)
    for link in links:
        if str(link['cid']) in compounds:
            herb_links[link['herb_id']].append(link)
    channels=sorted(t+'_'+endpoint for t in protocol['data']['targets'] for endpoint in ('Ki','Kd'))
    bundle={'schema':'dmease-molecular-evidence-v5','protocol_sha256':sha(OUT/'protocol.json'),
        'graph_fingerprint':read(OUT/'identifier_links.json')['evidence_fingerprint'],
        'affinity_models_sha256':sha(OUT/'affinity_models.json'),
        'source_files':{p.name:sha(p) for p in (OUT/'identifier_links.json',OUT/'structures.json',OUT/'identity_audit.json',OUT/'affinity_evaluation.json')},
        'targets':protocol['data']['targets'],'channels':channels,'herbs':herb_names,'herb_compounds':dict(herb_links),
        'compounds':compounds,'affinities':measurements+predictions,
        'scope':protocol['scope'],'aggregation':protocol['molecular_score'],
        'audit':{'resolved_herb_compound_links':sum(map(len,herb_links.values())), 'unique_compounds':len(compounds),
                 'measured_compound_target_endpoint_records':len(measurements),'predicted_compound_target_endpoint_records':len(predictions),
                 'unknown_compound_target_endpoint_records':len(unsupported),'measured_by_channel':dict(Counter(r['target']+'_'+r['endpoint'] for r in measurements)),
                 'predicted_by_channel':dict(Counter(r['target']+'_'+r['endpoint'] for r in predictions))}}
    bundle['fingerprint']=canonical_hash(bundle);save(OUT/'molecular_bundle.json',bundle)
    save(OUT/'unsupported_affinities.json',unsupported)
    save(OUT/'build_stamp.json',{'completed_at':datetime.now(timezone.utc).isoformat(),
        'source_lf_sha256':{p:hashlib.sha256((ROOT/p).read_text(encoding='utf8').encode()).hexdigest() for p in
         ['scripts/chemistry_v5.py','scripts/affinity_chain_v5.py','scripts/affinity_v4.py','src/dmease/affinity_research.py']}})
    print(bundle['audit'],flush=True)


if __name__=='__main__':main()

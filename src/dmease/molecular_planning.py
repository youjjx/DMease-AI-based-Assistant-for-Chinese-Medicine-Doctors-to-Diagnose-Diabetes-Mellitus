"""Molecular evidence is part of the policy state, action features and reward.

Scores retrieve evidence, not prescriptions: no action direction, concentration,
exposure or clinical effect can be inferred from a constituent's affinity.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .evidence_policy import canonical_hash
from .planning_v4 import PlanningEnvironment


class MolecularEvidence:
    def __init__(self, path, graph):
        self.data=json.loads(Path(path).read_text(encoding='utf8'))
        data=dict(self.data); fingerprint=data.pop('fingerprint')
        if canonical_hash(data)!=fingerprint or data['schema']!='dmease-molecular-evidence-v5':
            raise ValueError('分子证据文件校验失败。')
        if data['graph_fingerprint']!=graph.fingerprint:
            raise ValueError('分子证据与当前知识图谱版本不匹配。')
        self.fingerprint=fingerprint;self.graph=graph;self.channels=data['channels']
        self.channel_index={name:i for i,name in enumerate(self.channels)}
        self.compound_records=defaultdict(list)
        for row in data['affinities']:
            if row['kind'] not in ('measured','predicted') or not np.isfinite(row['index']) or not 0<=row['index']<=1:
                raise ValueError('非法分子证据记录。')
            self.compound_records[row['cid']].append(row)

    def matrices(self, mode='combined'):
        if mode not in ('combined','measured_only'):
            raise ValueError('Unknown molecular evidence mode')
        values=np.zeros((self.graph.action_dim,len(self.channels)),dtype=np.float32)
        known=np.zeros_like(values,dtype=bool);winners={}
        for herb_id,links in self.data['herb_compounds'].items():
            if herb_id not in self.graph.hidx:continue
            i=self.graph.hidx[herb_id]
            for link in links:
                for row in self.compound_records[link['cid']]:
                    if mode=='measured_only' and row['kind']!='measured':continue
                    j=self.channel_index[row['target']+'_'+row['endpoint']]
                    # Max avoids counting duplicate aliases or repeated herb-compound paths.
                    if not known[i,j] or row['index']>values[i,j]:
                        values[i,j]=row['index'];winners[i,j]={'link':link,'affinity':row}
                    known[i,j]=True
        return values,known,winners

    def trace(self, herb_index, mode='combined'):
        _,_,winners=self.matrices(mode);output=[]
        for (i,j),record in winners.items():
            if i!=herb_index:continue
            link=record['link'];affinity=record['affinity'];structure=self.data['compounds'][str(link['cid'])]
            output.append({'herb':self.graph.herbs[i]['name'],'channel':self.channels[j],
                'tcmsp_id':link['tcmsp_id'],'symmap_ids':link['symmap_ids'],'cid':link['cid'],
                'compound_name':link['compound_name'],'smiles':structure['parent_smiles'],
                'inchikey':structure['parent_inchikey'],'pubchem_url':structure['source_url'],
                'herb_source_url':link['herb_source_url'],**affinity})
        return output


class MolecularEnvironment(PlanningEnvironment):
    def __init__(self, graph, task, chemistry, weight=.35, mode='combined'):
        self.chemistry=chemistry;self.mode=mode
        if not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('分子证据权重必须在0至1之间。')
        self.weight=float(weight)
        requested=task.get('molecular_targets',list(chemistry.data['targets']))
        if not requested or set(requested)-set(chemistry.data['targets']):raise ValueError('请选择已配置的研究靶点。')
        self.requested=np.array([name.rsplit('_',1)[0] in requested for name in chemistry.channels],dtype=bool)
        self.affinity,self.known,self.winners=chemistry.matrices(mode)
        self.available=self.requested & self.known.any(0)
        self.effective_weight=self.weight if self.available.any() else 0.
        super().__init__(graph,task)
        molecular_useful=(self.affinity[:,self.available]>0).any(1)
        self.allowed=self.base.allowed & (self.useful | molecular_useful)
        self.observation_dim+=len(chemistry.channels)*3+2
        self.feature_dim+=len(chemistry.channels)*2

    def molecular_coverage(self, selected):
        chosen=self.vector(selected).astype(bool)
        values=self.affinity[chosen].max(0) if chosen.any() else np.zeros(len(self.chemistry.channels))
        known=self.known[chosen].any(0) if chosen.any() else np.zeros(len(self.chemistry.channels),dtype=bool)
        score=float(values[self.available].mean()) if self.available.any() else 0.
        return values,known,score

    def _score(self, selected):
        _,_,m=self.base.coverage(self.vector(selected));_,_,molecular=self.molecular_coverage(selected)
        w=self.effective_weight
        objective=(1-w)*m['objective']+w*molecular-w*.02*int(selected).bit_count()
        return (float(objective),m['symptom_evidence_coverage'],m['target_evidence_coverage'],molecular)

    def state(self):
        observation,features,mask=super().state()
        values,known,_=self.molecular_coverage(self.selected)
        extra=np.r_[values,known,self.requested,self.effective_weight,float(self.mode=='measured_only')].astype(np.float32)
        changes=np.maximum(self.affinity-values,0)*self.available
        # Unknown zero and observed/predicted zero have distinct availability bits.
        action_features=np.c_[changes,self.known*self.requested].astype(np.float32)
        action_features=np.r_[action_features,np.zeros((1,action_features.shape[1]),dtype=np.float32)]
        return torch.cat([observation,torch.from_numpy(extra)]),torch.cat([features,torch.from_numpy(action_features)],1),mask

    def exact_table(self):
        ids=np.flatnonzero(self.allowed);n=len(ids);all_masks=[];all_scores=[];g=self.graph;e=self.base
        tw=.3 if e.relevant.any() else 0;pw=.05 if len(e.relevant_edges) else 0
        for start in range(0,1<<n,4096):
            integers=np.arange(start,min(start+4096,1<<n),dtype=np.int64)
            bits=((integers[:,None]>>np.arange(n))&1).astype(np.float32)
            bits=bits[bits.sum(1)<=self.budget]
            selected=np.zeros((len(bits),self.h),dtype=np.float32);selected[:,ids]=bits
            for a,b in zip(*np.where(np.triu(self.pairs,1))):selected=selected[~((selected[:,a]>0)&(selected[:,b]>0))]
            if not len(selected):continue
            targets=(selected@g.affiliation>0).astype(np.float32)
            direct=(selected@g.direct.T>0).astype(np.float32)
            indirect=(targets@g.symptom_target.T)/np.maximum(g.symptom_target.sum(1),1)
            symptom=np.maximum(direct,indirect)@e.initial/max(float(e.initial.sum()),1e-8)
            target=targets@e.relevant/max(float(e.relevant.sum()),1)
            edges=e.relevant_edges
            ppi=(targets[:,edges[:,0]]*targets[:,edges[:,1]]).mean(1) if len(edges) else 0
            old=(.65*symptom+tw*target+pw*ppi)/(.65+tw+pw)-.02*selected.sum(1)
            molecular=np.zeros(len(selected))
            for channel in np.flatnonzero(self.available):
                molecular+=(selected*self.affinity[:,channel]).max(1)/self.available.sum()
            w=self.effective_weight;scores=(1-w)*old+w*molecular-w*.02*selected.sum(1)
            masks=(selected.astype(np.int64)*(1<<np.arange(self.h))).sum(1)
            all_masks.append(masks);all_scores.append(scores)
        return np.concatenate(all_masks),np.concatenate(all_scores)

    def explain(self):
        values,known,molecular=self.molecular_coverage(self.selected)
        return {'selected':[self.graph.herbs[i]['name'] for i in self.actions],
                'objective':self.score(self.selected)[0],'molecular_index':molecular,
                'requested_targets':self.task.get('molecular_targets',list(self.chemistry.data['targets'])),
                'unavailable_channels':[self.chemistry.channels[i] for i in np.flatnonzero(self.requested & ~self.available)],
                'channels':[{'channel':name,'requested':bool(self.requested[i]),'available_in_library':bool(self.available[i]),
                             'selected_evidence_available':bool(known[i]),'index':float(values[i]) if known[i] else None}
                            for i,name in enumerate(self.chemistry.channels)],
                'traces':[row for i in self.actions for row in self.chemistry.trace(i,self.mode)
                          if self.requested[self.chemistry.channel_index[row['channel']]]],
                'weight':self.weight,'effective_weight':self.effective_weight,'mode':self.mode,
                'evidence_fingerprint':self.chemistry.fingerprint,'scope':self.chemistry.data['scope']}

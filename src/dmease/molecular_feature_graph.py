"""Compact model-input snapshot for offline reproduction, not a clinical KG dump."""
import json
from pathlib import Path

import numpy as np

from .evidence_policy import canonical_hash, EvidenceGraph


class PlanningFeatureGraph:
    def __init__(self,path):
        data=json.loads(Path(path).read_text(encoding='utf8'));expected=data.pop('snapshot_fingerprint')
        if canonical_hash(data)!=expected or data['schema']!='dmease-planning-feature-snapshot-v5':
            raise ValueError('Planning feature snapshot checksum mismatch')
        self.fingerprint=data['original_evidence_fingerprint']
        self.symptoms=data['symptoms'];self.herbs=data['herbs'];self.target_ids=data['target_ids']
        self.sidx={x['id']:i for i,x in enumerate(self.symptoms)};self.hidx={x['id']:i for i,x in enumerate(self.herbs)}
        self.snames={x['name']:i for i,x in enumerate(self.symptoms)};self.hnames={x['name']:i for i,x in enumerate(self.herbs)}
        self.tidx={x:i for i,x in enumerate(self.target_ids)}
        for name in ('direct','symptom_target','affiliation'):setattr(self,name,np.array(data[name],dtype=np.float32))
        self.ppi=[tuple(x) for x in data['ppi']];self.action_dim=len(self.herbs)
        self.state_dim=2*len(self.symptoms)+2*len(self.target_ids)+2*self.action_dim
        self.signature=data['original_signature']
        self.data={'targets':data['targets'],**data.get('application_view',{})}
        self.origin_mode='published_public_feature_snapshot'

    def vector(self,symptoms):return EvidenceGraph.vector(self,symptoms)

    def paths(self,symptom_names,herb_index):return EvidenceGraph.paths(self,symptom_names,herb_index)


def export(graph,path):
    data={'schema':'dmease-planning-feature-snapshot-v5','original_evidence_fingerprint':graph.fingerprint,
          'original_signature':graph.signature,'symptoms':graph.symptoms,'herbs':graph.herbs,
          'target_ids':graph.target_ids,'targets':{t:graph.data['targets'][t] for t in graph.target_ids},
          'ppi':graph.ppi,'scope':'Computed feature view of the public graph used by the models; omits unrelated database rows and clinical narratives.'}
    data['application_view']={
        'syndromes':graph.data['syndromes'],'symptom_syndrome':graph.data['symptom_syndrome'],
        'symptom_herb':graph.data['symptom_herb'],
        'symptom_target':[r for r in graph.data['symptom_target'] if r['eligible'] and r['target'] in graph.tidx],
        'herb_target':[r for r in graph.data['herb_target'] if r['eligible'] and r['target'] in graph.tidx],
        'data_scope':'发布范围的真实公开证据特征快照：24症状、19药材、111证候及49个共享显著靶点；由原官方数据构建，非完整数据库，非患者临床证据。'}
    for name in ('direct','symptom_target','affiliation'):data[name]=getattr(graph,name).tolist()
    data['snapshot_fingerprint']=canonical_hash(data)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')


def published_service(snapshot,database,checkpoint):
    from .evidence_service import PublicEvidenceService,PublicGraphView
    from .evidence_policy import EvidencePolicy
    from .patient_db import PatientDatabase
    from .symptom_parser import SymptomParser
    class PublishedEvidenceService(PublicEvidenceService):
        def __init__(self):
            self.evidence=PlanningFeatureGraph(snapshot)
            if 'syndromes' not in self.evidence.data:raise ValueError('发布快照缺少界面所需证候关系。')
            self.graph=PublicGraphView(self.evidence);self.parser=SymptomParser(self.graph.symptoms)
            self.recommender=EvidencePolicy(self.evidence,checkpoint);self.database=PatientDatabase(database)
    return PublishedEvidenceService()

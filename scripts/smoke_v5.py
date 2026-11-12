"""Run the actual Streamlit workflow with nonclinical engineering inputs."""
import json
import argparse
import os
from pathlib import Path
import tempfile
import time

import torch
from streamlit.testing.v1 import AppTest

from dmease.auth import AuthStore
from dmease.patient_db import PatientDatabase

ROOT=Path(__file__).resolve().parents[1]


def widget(elements,label):return next(x for x in elements if x.label==label)


def healthy(app):
    assert not app.exception,[e.message for e in app.exception]
    assert not app.error,[e.value for e in app.error]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);report={'scope':'End-to-end UI check; inputs are engineering fixtures, not clinical cases.',
                                   'published_snapshot':args.snapshot}
    with tempfile.TemporaryDirectory(prefix='dmease-v5-ui-') as temp:
        os.environ['DMEASE_DATA_DIR']=temp
        os.environ['DMEASE_EVIDENCE_BUNDLE']=str(ROOT/('experiments/v5/planning_features.json' if args.snapshot else 'data/processed/evidence_bundle.json'))
        AuthStore(Path(temp)/'accounts.db').bootstrap('operator','Test-password1234')
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180)
        app.run();healthy(app)
        widget(app.text_input,'账号').set_value('operator');widget(app.text_input,'密码').set_value('Test-password1234')
        # Use actual login, rather than injecting an authenticated session.
        login=next(b for b in app.button if '登录' in b.label)
        login.click().run();healthy(app)
        app.sidebar.radio[0].set_value('研究复现').run()
        widget(app.radio,'研究工作区').set_value('成分结构与分子推荐').run();healthy(app)
        widget(app.selectbox,'查询药材').set_value('黄芪').run();healthy(app)
        widget(app.selectbox,'查询成分').set_value(5280934).run();healthy(app)
        report['structure_view']=True
        widget(app.selectbox,'分子研究项目').set_value('推荐与解释').run()
        widget(app.multiselect,'排除药材').set_value(['丹参'])
        widget(app.text_area,'配对排除（每行两味药材，用顿号分隔）').set_value('黄芪、山茱萸')
        widget(app.slider,'本次候选上限').set_value(4)
        widget(app.button,'执行分子链路推荐').click().run();healthy(app)
        signature,result=app.session_state['v5_molecular_result']
        assert result['model']['iteration']==100 and result['used']=='network'
        assert '丹参' not in result['selected'] and not {'黄芪','山茱萸'}<=set(result['selected']) and len(result['selected'])<=4
        assert all(s['policy_probability'] is not None for s in result['steps'])
        report['network_steps']=len(result['steps']);report['model']=result['model']
        widget(app.slider,'分子证据权重').set_value(.0).run();healthy(app)
        assert not any(m.label=='综合检索目标' for m in app.metric)  # Changed inputs invalidate visible results.
        widget(app.selectbox,'分子研究项目').set_value('实验结果').run();healthy(app)
        app.sidebar.radio[0].set_value('辅助分析').run();healthy(app)
        widget(app.text_input,'患者标识 *').set_value('分子链路工程测试')
        widget(app.text_area,'主诉与症状描述 *').set_value('工程测试文本：口渴，乏力，尿频。')
        widget(app.text_input,'已知过敏').set_value('黄芪')
        widget(app.multiselect,'本次明确禁用中药').set_value(['丹参'])
        widget(app.text_area,'本次配伍排除').set_value('山药、山茱萸')
        widget(app.selectbox,'选择策略').set_value('分子KAN/PPO＋贪心核对')
        widget(app.slider,'候选中药数量上限').set_value(4)
        widget(app.button,'开始辅助分析').click().run();healthy(app)
        widget(app.radio,'分析步骤').set_value('3 候选与解释').run();healthy(app)
        analysis=app.session_state['analysis_result'];molecular=analysis.metadata['molecular_evidence']
        chosen={r.name for r in analysis.recommendations}
        assert not chosen&{'黄芪','丹参'} and not {'山药','山茱萸'}<=chosen
        assert molecular['objective']>=molecular['guard_comparison']['greedy_objective']-1e-7
        widget(app.radio,'分析步骤').set_value('4 约束与保存').run();healthy(app)
        widget(app.checkbox,'已核对本次输入、约束提示与结果使用范围').check()
        widget(app.button,'保存本次分析').click().run();healthy(app)
        db=PatientDatabase(Path(temp)/'patients.db');record_id=app.session_state['saved_id']
        record=db.get(record_id,owner='operator')
        assert record['metadata']['molecular_evidence']['evidence_fingerprint']==molecular['evidence_fingerprint']
        assert record['metadata']['molecular_evidence']['traces']==molecular['traces']
        report['patient_record_persisted']=True;report['saved_candidate_count']=len(chosen)
        app.sidebar.radio[0].set_value('患者记录').run();healthy(app)
        widget(app.button,'复用资料并重新分析').click().run();healthy(app)
        assert 'analysis_result' not in app.session_state
        report['reuse_invalidates_result']=True
        widget(app.button,'退出登录').click().run();healthy(app)
        assert 'v5_molecular_result' not in app.session_state and 'analysis_result' not in app.session_state
        report['logout_clears_results']=True
    path=ROOT/('experiments/v5/ui_verification_snapshot.json' if args.snapshot else 'experiments/v5/ui_verification.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

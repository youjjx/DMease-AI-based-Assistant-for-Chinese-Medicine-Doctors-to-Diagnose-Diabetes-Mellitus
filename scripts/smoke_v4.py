"""Exercise the actual application with public evidence and an explicitly synthetic UI fixture."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from streamlit.testing.v1 import AppTest

from dmease.auth import AuthStore

ROOT = Path(__file__).resolve().parents[1]


def widget(elements, label):
    return next(x for x in elements if x.label == label)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--skip-llm', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(1)
    results = {'scope': 'UI smoke check with public graph; text input is an engineering fixture, not a clinical evaluation case.'}
    with tempfile.TemporaryDirectory(prefix='dmease-v4-ui-') as temp:
        os.environ['DMEASE_DATA_DIR'] = temp
        os.environ['DMEASE_EVIDENCE_BUNDLE'] = str(ROOT/'data/processed/evidence_bundle.json')
        AuthStore(Path(temp)/'accounts.db').bootstrap('operator', 'Test-password1234')
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=180)
        for key, value in dict(account='operator', credential_revision=1, last_activity=time.time()).items():
            app.session_state[key] = value
        app.run(); assert not app.exception
        app.sidebar.radio[0].set_value('研究复现').run()
        widget(app.radio, '研究工作区').set_value('结构化抽取与约束规划').run()
        sections = ['抽取验证', '规划对照', '外部辨证与亲和度', '原文抽取核对', '约束规划试算']
        for section in sections:
            widget(app.selectbox, '第四轮研究项目').set_value(section).run()
            assert not app.exception, section
        widget(app.multiselect, '排除药材').set_value(['麦冬'])
        widget(app.multiselect, '不得同时出现的两个药材（可留空）').set_value(['黄芪', '山药'])
        widget(app.slider, '候选数量上限').set_value(4)
        widget(app.button, '比较规划结果').click().run()
        assert not app.exception and not app.error
        _, planning = app.session_state['v4_planning_result']
        for row in planning['rows']:
            assert '麦冬' not in row['herbs'] and len(row['herbs']) <= 4
            assert not {'黄芪', '山药'} <= set(row['herbs'])
            if row['method'].endswith('greedy_guard'):
                assert row['objective'] >= planning['rows'][0]['objective']-1e-7
        results['views'] = sections; results['planning_methods'] = [r['method'] for r in planning['rows']]
        if not args.skip_llm:
            widget(app.selectbox, '第四轮研究项目').set_value('原文抽取核对').run()
            source = '工程测试文本：口渴，乏力，无胸痛。'
            widget(app.text_area, '输入待核对原文').set_value(source)
            widget(app.button, '执行原文约束抽取').click().run(timeout=240)
            assert not app.exception and not app.error
            _, extraction = app.session_state['v4_extraction_result']
            assert extraction['items']
            for item in extraction['items']:
                assert source[item['start']:item['end']] == item['text']
                assert all(source[m['start']:m['end']] == item['text'] for m in item['occurrences'])
            assert not any(q['invalid_json'] or q['invalid_schema'] for q in extraction['quality'])
            results['extraction'] = {'accepted_items': len(extraction['items']), 'seconds': extraction['seconds'], 'all_quotes_verified': True}
        widget(app.button, '退出登录').click().run()
        assert not app.exception and 'v4_planning_result' not in app.session_state and 'v4_extraction_result' not in app.session_state
    path = ROOT/'outputs/research_v4/ui_smoke.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

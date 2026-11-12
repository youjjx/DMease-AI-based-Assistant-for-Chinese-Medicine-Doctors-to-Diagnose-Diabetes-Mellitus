"""Fourth-round research views and explicit, inspectable optional tools."""
import hashlib
import json
from pathlib import Path

import streamlit as st


METHODS = {'greedy': '单步贪心', 'lookahead2': '双步搜索', 'greedy_search': '贪心＋集合优化', 'exact': '精确最优',
           'residual_kan': '增益先验＋KAN', 'residual_mlp': '增益先验＋MLP', 'kan_without_prior': 'KAN无加性先验',
           'rules24': '24症状规则', 'symmap_literal': 'SymMap词表匹配', 'copy_clauses': '整句切分对照',
           'simple_free': '简化提示词', 'schema': '字段及输入字符表约束', 'source_schema': '字段及原文子串约束',
           'v3_strict': '原四字段提示词'}


def label(method):
    for suffix, ending in (('_guarded', '＋贪心兜底'), ('_search', '＋集合优化'), ('_warm', '（PPO前）')):
        if method.endswith(suffix) and method not in METHODS:
            return METHODS.get(method[:-len(suffix)], method[:-len(suffix)]) + ending
    return METHODS.get(method, method)


@st.cache_resource
def extractor(path):
    from .v4_runtime import GroundedExtractor
    return GroundedExtractor(path)


def show(service, root):
    root = Path(root); folder = root/'experiments/v4'
    def read(name):
        return json.loads((folder/f'{name}.json').read_text(encoding='utf8'))
    if (folder/'planning_protocol.json').exists() and read('planning_protocol')['evidence_fingerprint'] != service.evidence.fingerprint:
        st.warning('当前证据版本与第四轮报告不同，以下指标仅对应报告的冻结版本。新模型试算需要匹配的证据包。')
    section = st.selectbox('第四轮研究项目', ['抽取验证', '规划对照', '外部辨证与亲和度', '原文抽取核对', '约束规划试算'])
    if section == '抽取验证':
        if not (folder/'extraction_test.json').exists():
            st.info('验证与测试结果生成中。'); return
        data = read('extraction_test'); selected = read('extraction_selection')['selected']
        st.caption('作者原有50条测试病例；直接对照专家症状短语。所有失败输出计入评价，词典投影不作为主指标。')
        st.write('验证集预先选中的方法：'+label(selected))
        st.dataframe([{'方法': label(m), '病例': r['records'], '精确率': r['surface']['precision'], '召回率': r['surface']['recall'],
                       'F1': r['surface']['f1'], 'JSON失败分段': r['invalid_json_chunks'], '结构失败分段': r['invalid_schema_chunks'],
                       '字符定位F1': r['characters']['f1'], '耗时中位数（秒）': r['median_seconds']}
                      for m, r in data['summary'].items()], hide_index=True, width='stretch')
        st.info('短语严格匹配会漏计同义表达；原文可定位比例和字符指标见报告。格式有效、引用存在，都不能单独证明医学判断正确。')
    elif section == '规划对照':
        if not (folder/'planning_summary.json').exists():
            st.info('重复训练与统一测试进行中。'); return
        data = read('planning_summary')
        profile = st.selectbox('查询范围', ['all', 'standard', 'requested_constraints'],
                               format_func=lambda x: {'all': '全部256条请求', 'standard': '常规检索请求', 'requested_constraints': '带排除、配对和数量限制的请求'}[x])
        st.dataframe([{'方法': label(m), '综合证据目标': r[profile]['objective'], '距精确最优差值': r[profile]['mean_gap_to_optimal'],
                       '达到最优比例': r[profile]['optimal_rate'], '约束违规': r[profile]['hard_violations'],
                       '单次耗时中位数（毫秒）': r[profile]['median_rollout_ms']} for m, r in data['summary'].items()],
                      hide_index=True, width='stretch')
        st.caption('128组症状，每组两个工程约束配置；训练、验证与测试按症状组隔离。三种网络各3个种子。约束配置不是临床禁忌标注，目标为公开证据覆盖。')
        st.write('验证选中的PPO迭代（−1为初始先验，0为监督预训练）：', {label(k): v for k, v in data['selected_iterations'].items()})
        with st.expander('查看原始网络相对贪心的配对统计'):
            st.json(data['raw_vs_greedy_base_clustered'])
        if (folder/'gating_summary.json').exists():
            gating = read('gating_summary')
            st.write('补充：验证集选择的保守改动阈值')
            st.dataframe([{'方法': label(m), '证据目标': r[profile]['objective'], '约束违规': r[profile]['hard_violations'],
                           '偏离贪心次数': r[profile]['overrides']} for m, r in gating['summary'].items()], hide_index=True, width='stretch')
            st.caption('网络评分差超过验证阈值才改动贪心动作。评分差不是概率；“贪心兜底”仍是额外运行后择优。')
    elif section == '外部辨证与亲和度':
        diagnosis = read('diagnosis_test')
        st.caption('冻结上一轮分类器，使用作者50条测试病例及独立专家答案，不按选择题选项缩小预测范围。')
        st.dataframe([{'模型': '原模型' if k == 'frozen_unweighted_v2' else '类别加权模型', '记录': v['groups']['all']['records'],
                       '首选命中': v['groups']['all']['top1'], '前三命中': v['groups']['all']['top3'],
                       '标签覆盖上限': v['groups']['all']['known_label_ceiling']} for k, v in diagnosis['results'].items()], hide_index=True, width='stretch')
        affinity = read('affinity_robustness')
        st.dataframe([{'隔离方式': '按文献' if k == 'document' else '按文献和骨架', '终点': name,
                       '训练': v['counts']['train'], '验证': v['counts']['validation'], '测试': v['counts']['test'],
                       '状态': '已评价' if v['status'] == 'evaluated' else '固定划分样本不足'}
                      for k, g in affinity['groupings'].items() for name, v in g['strata'].items()], hide_index=True, width='stretch')
        st.caption('严格文献隔离检查使用相同实测快照，不改划分凑样本，也不把样本不足计作模型失败。')
        loco = read('affinity_loco')
        st.write('逐来源组留出：固定超参数的探索性补充')
        st.dataframe([{'隔离方式': kind, '终点': name, '评价记录': r['evaluated_rows'], '总记录': r['rows'],
                       '评价来源组': r['evaluated_components'],
                       '均值基线MAE': r.get('pooled', {}).get('training_mean', {}).get('MAE_pK'),
                       'Ridge MAE': r.get('pooled', {}).get('ridge_alpha10', {}).get('MAE_pK')}
                      for kind, strata in loco['groupings'].items() for name, r in strata.items()], hide_index=True, width='stretch')
        st.caption('超参数沿用上一轮重叠数据的选择，训练集在折间重叠；这是探索性检查。未达汇总样本要求时MAE留空。')
    elif section == '原文抽取核对':
        st.caption('本地模型逐字抽取，保留存在性、原文位置和词表映射；结果供人工核对。')
        source = st.text_area('输入待核对原文', max_chars=3000, height=160, key='v4_extraction_source')
        signature = hashlib.sha256(source.encode()).hexdigest()
        model_path = root/'outputs/research_v3/local_model'
        if not (model_path/'verified_model.json').exists():
            st.info('本机尚未安装并校验研究用语言模型。请按项目说明准备固定版本权重。'); return
        if st.button('执行原文约束抽取', disabled=not source.strip()):
            progress = st.progress(0.)
            try:
                with st.spinner('本地逐段抽取中，首次加载需要较长时间'):
                    result = extractor(str(model_path)).extract(source, service.parser, progress.progress)
                st.session_state['v4_extraction_result'] = (signature, result)
            except (ValueError, OSError, ImportError, RuntimeError) as exc:
                st.error(f'本次抽取未完成：{exc}')
        stored = st.session_state.get('v4_extraction_result')
        if stored and stored[0] == signature:
            result = stored[1]
            st.dataframe([{'原文短语': r['text'], '模型判断': '存在' if r['present'] else '否认或已消失',
                           '首处起始位置（从0计）': r['start'], '首处结束位置（不含）': r['end'], '分段': r['segment'],
                           '本段匹配处数': len(r['occurrences']), '原文上下文': ' / '.join(x['context'] for x in r['occurrences']),
                           '核对提示': '；'.join(r['review_notes'])} for r in result['items']],
                         hide_index=True, width='stretch')
            st.write('标准症状映射', result['normalized'])
            st.dataframe([{'分段': q['segment'], '格式完整': not(q['invalid_json'] or q['invalid_schema']),
                           '达到输出上限': q['output_limit'], '未采用条目数': sum(q['rejected'].values())} for q in result['quality']],
                         hide_index=True, width='stretch')
            st.caption(result['scope'])
            st.download_button('下载本次抽取结果', json.dumps(result, ensure_ascii=False, indent=2), 'dmease_grounded_extraction.json', 'application/json')
    else:
        st.caption('在同一目标下直接比较模型、贪心、双步搜索与精确解，显示每种方法实际选出的药材。人工指定的组合限制仅用于研究请求。')
        names = st.multiselect('症状组合', service.graph.symptom_names, default=[n for n in ('口渴', '乏力', '尿频') if n in service.graph.symptom_names])
        excluded = st.multiselect('排除药材', service.graph.herb_names)
        pair = st.multiselect('不得同时出现的两个药材（可留空）', service.graph.herb_names, max_selections=2)
        budget = st.slider('候选数量上限', 1, min(8, len(service.graph.herb_names)), 4)
        variant = st.selectbox('对照网络', ['residual_kan', 'residual_mlp', 'kan_without_prior'], format_func=label)
        seed = st.selectbox('重复训练编号', [42, 43, 44])
        task = {'symptoms': dict.fromkeys(names, 1.), 'forbidden': excluded, 'forbidden_pairs': [pair] if len(pair) == 2 else [], 'budget': budget}
        signature = json.dumps([task, variant, seed, service.evidence.fingerprint], sort_keys=True)
        if st.button('比较规划结果', disabled=not names):
            try:
                from .v4_runtime import planning_comparison
                with st.spinner('执行相同约束下的规划对照'):
                    result = planning_comparison(service.evidence, task, root, variant, seed)
                st.session_state['v4_planning_result'] = (signature, result)
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(f'本次试算未完成：{exc}')
        stored = st.session_state.get('v4_planning_result')
        if stored and stored[0] == signature:
            result = stored[1]
            iteration = result['selected_iteration']
            stage = '初始增益先验' if iteration == -1 else '监督预训练' if iteration == 0 else f'PPO第{iteration}次更新'
            st.write('本组原始网络的验证选中阶段：'+stage)
            gate = next((r.get('configuration') for r in result['rows'] if r['method'] == 'gated_network'), None)
            if gate:
                st.write('保守阈值方案采用：'+{'greedy': '纯贪心', 'warm': '监督预训练快照', 'selected': '原验证选中快照'}[gate['mode']],
                         '；改动阈值：', gate['margin'])
            st.dataframe([{'方法': {'network': '原始网络', 'network_with_greedy_guard': '网络与贪心择优',
                                    'gated_network': '保守阈值网络', 'gated_with_greedy_guard': '保守阈值网络与贪心择优'}.get(r['method'], label(r['method'])),
                           '候选药材': '、'.join(r['herbs']), '证据目标': r['objective'], '秒': r['seconds']} for r in result['rows']],
                         hide_index=True, width='stretch')
            st.caption('“网络与贪心择优”额外运行贪心后比较目标值；其改善不能归因于网络本身。证据覆盖分数不是疗效概率。')
            st.download_button('下载规划对照结果', json.dumps(result, ensure_ascii=False, indent=2), 'dmease_planning_comparison.json', 'application/json')
    report = folder/'report.html'
    if report.exists():
        st.download_button('下载第四轮研究报告', report.read_bytes(), 'DMease_第四轮研究报告.html', 'text/html')

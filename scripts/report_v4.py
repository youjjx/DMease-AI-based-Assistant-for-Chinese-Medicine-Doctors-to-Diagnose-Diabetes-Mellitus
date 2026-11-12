"""Build a standalone scientific report exclusively from completed v4 artifacts."""
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import build_experiment_report as render
from dmease.research_v4_ui import label

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/v4'


def read(name):
    return json.loads((OUT/f'{name}.json').read_text(encoding='utf8'))


def main():
    extraction = read('extraction_test'); selection = read('extraction_selection')
    planning = read('planning_summary'); protocol = read('planning_protocol')
    diagnosis = read('diagnosis_test'); affinity = read('affinity_robustness')
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font)); plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'DMease-v4'})
    render.FIGURES = OUT/'figures'; render.FIGURES.mkdir(exist_ok=True)
    table, chart = render.table, render.chart
    body = ['<header><p>DMease · 第四轮研究</p><h1>结构化抽取、约束规划与外部验证</h1><p>真实模型推理 · 作者分区与专家参考 · 三种子重复训练 · 保留完整对照</p></header>',
            '<aside>更正：前几轮误将病例JSON中的空答案字段理解为作者未公开验证与测试答案。作者仓库实际在Results目录单独提供答案，evaluate.py也将其作为标准答案。'
            '本轮按病例ID完整对齐50条验证和50条测试记录，前200条训练记录不再作为新测试集。旧轮数值保持原样，关于答案不可取得的表述以本更正为准。'
            '建方案时检查过数据格式及少量测试答案，不能声称从未见过任何测试内容；没有使用测试模型成绩选提示词或模型。</aside>']
    fig, ax = plt.subplots(figsize=(14, 5), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 5)); ax.axis('off')
    for y, title, labels in ((3.8, '症状抽取', ['原文与分段位置', '约束解码\n短语＋存在性', '原文引用核验', '标准症状映射', '上下文人工核对']),
                             (1.5, '约束规划', ['症状＋完整限制\n剩余数量预算', '增益先验＋网络\n可选保守阈值', '可行候选或STOP', '更新覆盖与轨迹', '可选整条轨迹\n与贪心择优'])):
        ax.text(.05, y+.65, title, weight='bold', fontsize=12)
        for i, text in enumerate(labels):
            x = .95+i*2.0
            ax.text(x, y, text, ha='center', va='center', fontsize=11,
                    bbox={'boxstyle': 'round,pad=.5', 'facecolor': '#f1f4f6', 'edgecolor': '#333'})
            if i < 4:
                ax.annotate('', xy=(x+1.23, y), xytext=(x+.77, y), arrowprops={'arrowstyle': '->', 'color': '#333'})
    ax.plot([6.95, 6.95, .95], [1.05, .8, .8], color='#333', linewidth=1)
    ax.annotate('', xy=(.95, 1.1), xytext=(.95, .8), arrowprops={'arrowstyle': '->', 'color': '#333'})
    ax.text(4, .55, '每步更新状态并重新评分，直至STOP或达到数量上限', ha='center', va='top', fontsize=10)
    ax.text(.2, .08, '训练：精确续接监督＋PPO；验证：选快照与阈值；测试：冻结后统一评价。', fontsize=10)
    body.append(chart(fig, 'architecture', '抽取引用与医学解释分开；额外搜索或贪心择优与网络贡献分开记录。'))
    body.append('<h2>1. 语言模型抽取：把格式问题与医学语义分开</h2><p>旧提示词同时要求term、evidence、present、canonical，实际输出经常缺少字段或截断。'
                '本轮只让模型抽取逐字短语及存在性；规范症状映射独立处理。原文按标点分成最多240字的片段，保留偏移量，输出最多16项，每项最多32字。'
                '相同Qwen2.5-1.5B-Instruct、FP16、贪婪解码，固定768个输出token。比较自由生成、JSON字段约束、JSON加原文子串约束，均无重试或补括号。'
                '原四字段提示词保留384个输出token，作为整体旧配置对照；它与新配置存在提示词、分段和长度的共同变化，不能将差异全部归因于单一因素。</p>'
                '<p>实现细节：两种约束解码的可用字符表都是解析器默认ASCII字符加当前输入中的字符，以支持中文。'
                '因此schema同时受到字段类型与输入字符表限制，并非完全纯粹的格式消融；source_schema进一步限制为原文连续子串。'
                '此说明不更改已冻结代码、提示词或生成结果，详见extraction_implementation_notes.json。</p>'
                '<p>主要指标直接对照专家症状短语，含专家记录的阴性体征；存在性准确率没有完整独立金标准，未伪造该成绩。'
                '严格文字匹配不处理同义改写，因此另报可定位参考的字符覆盖F1，并保留整句复制对照。24症状词典投影仅作为次要接口指标。</p>')
    chosen = selection['selected']; selected = extraction['summary'][chosen]
    body.insert(1, f'<p><b>本轮主要结果：</b>验证选中的抽取方案在作者50例测试集上，严格专家短语F1为{selected["surface"]["f1"]:.4f}。'
                f'规划测试中，贪心目标为{planning["summary"]["greedy"]["all"]["objective"]:.5f}，'
                f'无加性先验KAN的选中快照为{planning["summary"]["kan_without_prior"]["all"]["objective"]:.5f}，'
                f'将该网络与贪心完整结果验算择优后为{planning["summary"]["kan_without_prior_guarded"]["all"]["objective"]:.5f}。'
                '结构化抽取已有实质改善，原始KAN/PPO仍未超过贪心；额外候选验算的收益单独报告。</p>')
    body.append(f'<p>验证集预先选中<b>{label(chosen)}</b>。测试集50例微平均F1为<b>{selected["surface"]["f1"]:.4f}</b>，'
                f'精确率{selected["surface"]["precision"]:.4f}、召回率{selected["surface"]["recall"]:.4f}；'
                f'{selected["chunks"]}个分段中JSON失败{selected["invalid_json_chunks"]}个，结构失败{selected["invalid_schema_chunks"]}个。'
                f'专家短语原文可逐字定位比例为{selected["gold_verbatim_fraction"]:.1%}，这限制了“严格短语＋原文约束”能达到的召回率。</p>')
    body.append(table(['方法', '短语精确率', '短语召回率', '短语F1', '字符F1', 'JSON失败/分段', '结构失败', '达长度上限', '耗时中位数/秒'],
                     [[label(m), *[f'{r["surface"][k]:.4f}' for k in ('precision', 'recall', 'f1')], f'{r["characters"]["f1"]:.4f}',
                       f'{r["invalid_json_chunks"]}/{r["chunks"]}' if r['chunks'] else '—', r['invalid_schema_chunks'], r['output_limits'], f'{r["median_seconds"]:.2f}']
                      for m, r in extraction['summary'].items()]))
    body.append('<p>规则与词表方法的耗时未计时，表中0不是零耗时。失败分段保留在整例分母中。原文子串约束只保证词语来自输入，'
                '仍可能误取年龄、治疗经过、不完整短语，或错误判断否定；不能把JSON成功率当作医学抽取准确率。</p>')
    names = list(extraction['summary']); fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    axes[0].barh([label(m) for m in names], [extraction['summary'][m]['surface']['f1'] for m in names], color='#406580')
    axes[0].invert_yaxis(); axes[0].set(xlim=(0, 1), xlabel='严格专家短语微平均F1', title='50例作者测试分区')
    model_names = ['v3_strict', 'simple_free', 'schema', 'source_schema']
    axes[1].barh([label(m) for m in model_names], [extraction['summary'][m]['cases_with_invalid_chunks'] for m in model_names], color='#777777')
    axes[1].invert_yaxis(); axes[1].set(xlim=(0, 50), xlabel='包含JSON或结构失败分段的病例数', title='格式失败与语义成绩分别报告')
    body.append(chart(fig, 'extraction', '所有方法使用同一50例；直接专家参考不依赖系统24症状词表。'))
    extraction_statistics = read('extraction_statistics')
    body.append(table(['验证选中方案对照', '短语F1差值', '95%病例重采样区间', 'Holm p'],
                     [[label(name), f'{r["micro_f1_difference"]:+.4f}', str([round(x, 4) for x in r['case_bootstrap_95ci']]),
                       f'{r["holm_p"]:.4f}'] for name, r in extraction_statistics['comparisons'].items()]))
    body.append('<p>固定比较验证选中的方案与词表、整句切分、旧提示词三种对照；按病例配对重采样10000次、病例内交换预测20000次，'
                '三次比较用Holm校正。这是文本抽取指标的不确定性分析，不代表临床诊断或疗效显著性。</p>')
    body.append('<h2>2. KAN/PPO：状态可见性、精确监督与强基线</h2><p>旧状态缺少完整的后续配对限制和剩余数量预算，动作空间还排除了暂时不增加目标的候选。'
                '新环境补全这些信息，显式增加STOP动作，只以硬约束屏蔽动作。目标函数保持公开证据覆盖与候选数量代价，未加入构造亲和度或疗效分数。</p>'
                '<p>枚举症状集合，排除前两轮全部症状组，按固定哈希分320/64/128组用于训练/验证/测试，每组包含常规与请求约束两种配置，合计640/128/256条。'
                '请求约束为工程测试，不是临床禁忌规则。训练阶段精确枚举可行集合，为每个动作提供最优后续目标监督；测试参考答案不参与训练。'
                '网络以当前增益为加性先验，再学习后续收益修正；另设同容量MLP和去掉加性先验的KAN。三种网络各跑42/43/44三个种子，'
                '50轮监督预训练、120次PPO更新。验证集可选择初始先验、预训练或PPO快照；全部9个模型冻结后统一测试。</p>')
    summary = planning['summary']; greedy = summary['greedy']['all']['objective']; exact = summary['exact']['all']['objective']
    body.append(f'<p>同一256条请求上，单步贪心均值{greedy:.6f}，精确最优{exact:.6f}，平均可改进空间为{exact-greedy:.6f}。'
                '这是任务内的目标值差距，不是临床效益差异。</p>')
    body.append(table(['方法', '全部目标', '常规目标', '约束目标', '距最优差值', '最优比例', '违规总数', '单次耗时中位数/毫秒'],
                     [[label(m), *[f'{r[p]["objective"]:.6f}' for p in ('all', 'standard', 'requested_constraints')],
                       f'{r["all"]["mean_gap_to_optimal"]:.6f}', f'{r["all"]["optimal_rate"]:.1%}', r['all']['hard_violations'],
                       f'{r["all"]["median_rollout_ms"]:.2f}'] for m, r in summary.items()]))
    body.append('<p>网络结果先在三个种子及每组两个约束配置间取平均，再以128个症状组进行配对重采样和符号翻转检验，'
                '三种原始网络对贪心比较使用Holm校正。种子数仅3，不能据此获得稳定的训练随机性分布。耗时包含整条决策轨迹的特征计算，'
                '部分实验并行运行，CPU调度会影响延迟，因此只作本机描述，非独立性能基准。</p>')
    body.append(table(['原始网络', '均值差（对贪心）', '95%组重采样区间', 'Holm p', '优/平/劣症状组', '验证选中迭代'],
                     [[label(m), f'{r["mean_difference"]:+.6f}', str([round(x, 6) for x in r['query_bootstrap_95ci']]),
                       f'{r["holm_p"]:.4f}', str(r['wins_ties_losses']), str(planning['selected_iterations'][m])]
                      for m, r in planning['raw_vs_greedy_base_clustered'].items()]))
    for variant in protocol['variants']:
        stats = planning['raw_vs_greedy_base_clustered'][variant]
        change = planning['warm_to_selected'][variant]['mean_difference']
        body.append(f'<p><b>{label(variant)}</b>：相对贪心差值{stats["mean_difference"]:+.6f}；验证选中快照相对PPO前模型的测试差值{change:+.6f}。'
                    f'验证选中迭代为{planning["selected_iterations"][variant]}（−1代表初始先验，0代表监督预训练）。'
                    '只有选中正数迭代且性能确有提升，才可能支持本配置中PPO的增益；仍需结合各种子和配对区间判断。</p>')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout='constrained')
    for ax, variant in zip(axes, protocol['variants']):
        for seed in protocol['seeds']:
            history = read(f'planning_training/{variant}_{seed}')['history']
            ax.plot([r['iteration'] for r in history], [r['objective'] for r in history], marker='.', label=str(seed))
        ax.set(title=label(variant), xlabel='迭代（−1初始，0预训练）', ylabel='验证目标'); ax.legend(title='种子')
    body.append(chart(fig, 'planning_validation', '所有验证快照均保留；没有按测试集重新选择迭代。各面板纵轴尺度不同。'))
    shown = ['greedy', 'lookahead2', 'greedy_search', 'residual_kan', 'residual_mlp', 'kan_without_prior', 'exact']
    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    ax.barh([label(m) for m in shown], [summary[m]['all']['objective']-greedy for m in shown], color='#406580')
    ax.axvline(0, color='black', linewidth=1); ax.invert_yaxis(); ax.set(xlabel='相对单步贪心的目标差', title='同一测试请求的原始方法与强基线')
    body.append(chart(fig, 'planning_comparison', '额外的贪心兜底和集合优化结果另列；它们不能被归因于KAN自身。'))
    diagnostics = read('planning_diagnostics')
    body.append('<h3>动作级错误定位</h3><p>以下只分析训练与验证状态，不再拟合或重新选模。对每个首动作，假设其后执行精确最优续接，'
                '计算相对最佳续接值的损失；这与完整策略轨迹的损失不同。统计网络破坏了多少个原本最优的贪心首动作，以及修复了多少个贪心非最优动作。</p>')
    body.append(table(['验证状态方法', '状态数', '原本贪心最优/非最优', '被改错的最优动作', '修复的非最优动作', '平均续接损失'],
                     [[name, r['states'], f'{r["greedy_optimal_first_actions"]}/{r["greedy_suboptimal_first_actions"]}',
                       r['damaged_greedy_optimal_actions'], r['rescued_greedy_suboptimal_actions'], f'{r["mean_best_continuation_regret"]:.6f}']
                      for name, r in diagnostics['results']['validation'].items()]))
    gating = read('gating_summary')
    body.append('<h3>保守改动阈值：动作误差分析后的补充方案</h3><p>在看到训练状态中“少量修复伴随大量误改”的现象后、尚未计算规划测试成绩前，'
                '另行固定此方案。网络首选动作相对其给贪心动作的评分优势超过阈值，才允许改动。分别考察预训练和已选快照，阈值固定为0、0.002、0.005、0.01、0.02、0.04、0.08，'
                '连同纯贪心共15个候选，仅在128条验证请求上选第一个最优。全部9个选择在任何规划测试前冻结。评分差不是校准后的正确概率，'
                '阈值选择也不保证逐查询优于贪心；额外整条轨迹择优才提供这一保证。</p>')
    body.append(table(['训练版本', '验证选中来源', '阈值', '验证目标'],
                     [[name, decision['candidates'][decision['selected_index']]['mode'],
                       decision['candidates'][decision['selected_index']]['margin'],
                       f'{decision["candidates"][decision["selected_index"]]["validation_objective"]:.6f}']
                      for name, decision in gating['selection']['selections'].items()]))
    body.append(table(['保守方案', '全部测试目标', '常规测试目标', '约束测试目标', '约束违规'],
                     [[label(name), *[f'{r[p]["objective"]:.6f}' for p in ('all', 'standard', 'requested_constraints')], r['all']['hard_violations']]
                      for name, r in gating['summary'].items()]))
    body.append(table(['保守方案对贪心', '均值差', '95%组区间', '合并6比较Holm p'],
                     [[label(name), f'{r["mean_difference"]:+.6f}', str([round(x, 6) for x in r['query_bootstrap_95ci']]),
                       f'{r["holm_p_six_comparisons"]:.4f}'] for name, r in gating['comparisons'].items()]))
    body.append('<p>此处合并原始3种网络和阈值3种网络形成6比较校正；原实验表中的3比较校正仍保留。'
                '该方案结合了网络与贪心先验，选中预训练模型时更不能归因为PPO增益。多阈值选择仍可能过拟合较小验证集，全部测试结果如实保留。</p>')
    body.append('<h2>3. 外部辨证与亲和度稳健性</h2><p>上一轮的两个148标签分类器保持权重、特征和预处理不变，'
                '用TCMEval专家的选项字母还原证候名称。模型在完整标签表预测，不读取选项来缩小答案空间；只去除空白和末尾“证”，没有按结果追加同义词。</p>')
    body.append(table(['分类器', '记录', '首选命中', '前三命中', '已知标签覆盖上限', '含已知金标准病例的首选命中'],
                     [[m, r['groups']['all']['records'], *[f'{r["groups"]["all"][k]:.1%}' for k in ('top1', 'top3', 'known_label_ceiling', 'conditional_top1')]]
                      for m, r in diagnosis['results'].items()]))
    body.append('<p>上述50条输入与TCM-SD训练及验证输入的规范化全文精确比较结果保存在diagnosis_test.json。精确文本不重叠不排除改写或同一患者关联。'
                '本实验仍显示标签体系迁移是主要限制，不应把少量首选命中变化解释为已达到论文诊断准确率。</p>'
                '<p>亲和度复用ChEMBL 37的468条标准化实测记录，新增全局来源文献连通分组，以及“文献＋骨架”联合连通分组；相同化合物跨终点也归在同组。'
                '按固定组哈希70/15/15划分，不为得到可评价结果而换随机种子。各终点仍要求训练≥40、验证≥15、测试≥15。</p>')
    body.append(table(['隔离方式', '终点', '训练', '验证', '测试', '状态'],
                     [[kind, name, *[r['counts'][s] for s in ('train', 'validation', 'test')], r['status']]
                      for kind, g in affinity['groupings'].items() for name, r in g['strata'].items()]))
    body.append('<p>固定划分未满足样本量时不计算模型成绩。第三轮PPARG-Ki的骨架留出结果仍有效地描述其原测试，但其训练与测试有8篇来源文献重叠，'
                '本轮不能据不足的文献隔离数据宣称已完成更强外部验证。需扩充独立实验来源，而非把关联数据库分数变为实测亲和度。</p>')
    loco = read('affinity_loco')
    body.append('<h3>补充：逐来源组留出的探索性检验</h3><p>固定三分区样本不足后，另立方案逐个留出来源连通组，其余组训练。'
                '不再选超参数，固定使用上一轮的Ridge α=10与各折训练均值。由于该超参数曾在重叠数据上被选择，本检验属于探索性稳健性分析，'
                '不是完全独立的新模型确认。训练少于40条的折跳过并公开覆盖率；仅在至少2个留出组、15条评价记录时汇总成绩。</p>')
    body.append(table(['隔离方式', '终点', '评价/总数', '评价来源组', '均值基线MAE', 'Ridge MAE', '差值95%组区间'],
                     [[kind, name, f'{r["evaluated_rows"]}/{r["rows"]}', r['evaluated_components'],
                       f'{r["pooled"]["training_mean"]["MAE_pK"]:.4f}' if r['status'] == 'evaluated' else '样本不足',
                       f'{r["pooled"]["ridge_alpha10"]["MAE_pK"]:.4f}' if r['status'] == 'evaluated' else '样本不足',
                       str([round(x, 4) for x in r['comparison']['ci95']]) if r['status'] == 'evaluated' else '—']
                      for kind, strata in loco['groupings'].items() for name, r in strata.items()]))
    body.append('<p>各折训练集互相重叠，区间仅为描述性组重采样；部分终点仅覆盖部分记录，不能外推到被跳过的大来源组。'
                '文献与骨架联合隔离的PPARG-Ki覆盖253条记录，可与本轮同折均值基线比较；它与第三轮32个骨架留出化合物的测试人群不同，不能直接相减归因。</p>')
    body.append('<h2>4. 软件改进与论文差距</h2><p>研究界面增加抽取评价、规划对照、外部辨证与亲和度表、原文抽取核对、约束规划试算。'
                '原文核对固定使用原文子串约束，显示逐字引用、存在性、偏移、重复位置、冲突提示和词表映射；实验选模结果单独呈现。'
                '规划显示各方法实际候选并保留约束。已发布9组训练的选中与预训练快照及校验值，语言模型权重需本地按固定版本安装。'
                '这些研究工具不自动写回患者诊断，也不把测试结果作为临床部署批准。</p>'
                '<ul><li>已补：结构化真实模型推理、独立专家短语测试、显式停止与完整约束状态、精确动作监督、PPO前后对照、三种子与MLP消融、强搜索及精确上限、外部辨证测试、文献隔离亲和度审计。</li>'
                '<li>仍缺：原论文108,746条门诊记录与50例糖尿病配对专家数据、经过核验的体质效应标签、药材成分到结构和实测亲和度的可靠连接、完整临床禁忌及临床效用评价。</li>'
                '<li>实验是公开证据实现的进一步验证，尚不能声称复现论文86%准确率、0.83 F1或0.67 TCS。软件功能完成度与临床效果证明应分别描述。</li></ul>')
    body.append('<h2>5. 来源与复现</h2><p><a href="https://github.com/zhuyan166/TCMEval/tree/12473b50ce94ad2fd22b1e984b826b0a618c4407/evaluation/TCMEval-SDT">TCMEval固定作者版本</a> · '
                '<a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct">Qwen模型</a> · <a href="https://github.com/noamgat/lm-format-enforcer">LM Format Enforcer</a> · '
                '<a href="https://github.com/Borororo/ZY-BERT">TCM-SD</a> · <a href="https://www.ebi.ac.uk/chembl/">ChEMBL</a></p>'
                '<p>各protocol文件固定设计；selection与training文件记录测试前选择；planning_results保存逐请求候选；extraction结果保存逐病例哈希及混淆计数，'
                '不公开病例原文和模型生成文本。运行scripts/verify_v4.py离线检查源代码校验值、分区、9个模型、全部分母、硬约束及配对统计。'
                '复现命令和数据许可见仓库README与DATA_LICENSE.txt。实验不需要GitHub账号口令。</p>')
    css = "body{font:16px/1.75 'Microsoft YaHei',sans-serif;max-width:1200px;margin:auto;padding:36px;color:#222}header{border-bottom:3px solid #333}h2{margin-top:42px}aside{padding:18px;background:#f2f2f2;border-left:4px solid #555}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #aaa;padding:7px;text-align:left}th{background:#eee}figure{margin:25px 0}img{max-width:100%}figcaption{color:#555;font-size:14px}a{color:#17597b}"
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DMease 第四轮研究报告</title><style>'+css+'</style><body>'+'\n'.join(body)+'</body></html>'
    (OUT/'report.html').write_text(document, encoding='utf8', newline='\n')
    print(OUT/'report.html')


if __name__ == '__main__':
    main()

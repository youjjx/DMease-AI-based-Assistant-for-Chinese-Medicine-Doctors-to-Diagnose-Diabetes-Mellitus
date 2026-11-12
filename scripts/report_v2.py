"""Render the complete second-round report from immutable numerical results."""
from pathlib import Path
import json
import html
import csv

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import build_experiment_report as render

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v2"
NAMES = {"greedy": "证据增益", "legacy": "旧版 RBF/KAN-PPO", "stable_rbf": "稳定 RBF 方案",
         "stable_spline": "三次样条 KAN", "oracle_spline": "最优集合示范 + 样条", "oracle_mlp": "最优集合示范 + MLP"}


def read(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf8"))


def main():
    summary, external, supervised, pathways = (read(k) for k in ("summary", "external_syndrome", "supervised_syndrome", "pathways"))
    render.FIGURES = OUT / "figures"; render.FIGURES.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "DMease-v2"})
    table, chart = render.table, render.chart
    result = summary["results"]
    legacy = result["legacy"]["raw"]["objective"]
    stable = result["stable_rbf"]["raw"]["objective"]
    greedy = result["greedy"]["raw"]["objective"]
    gap_reduction = 1 - (greedy-stable)/(greedy-legacy)
    body = ['<header><p>DMease · 第二轮可复现研究</p><h1>KAN/PPO诊断、优化与外部辨证评价</h1><p>固定新查询留出 · 真实公开标注数据 · 全部结果保留</p></header>']
    body.append('<aside>本报告区分三种证据：数据库查询优化、公开病例回顾性辨证、通路富集。它们不能互换，也不等于临床疗效验证。'
                '原论文的50例糖尿病配对病例与亲和度模型仍未取得。</aside>')
    body.append('<h2>1. 得到了什么改进</h2>'
                f'<p>在全新192个测试查询上，旧策略平均参考目标为{legacy:.6f}，稳定训练方案为{stable:.6f}，与贪心的差距缩小{gap_reduction:.1%}。'
                f'但贪心仍为{greedy:.6f}，当前结果不支持神经策略优于它。稳定RBF方案的三个种子均由验证集选择了PPO前的模型，'
                '因此这部分收益主要体现了预训练与检查点保留，不能归功于PPO优化。</p>'
                f'<p>统一的删除/增补/交换搜索把贪心目标提高到{result["greedy"]["shared_local_search"]["objective"]:.6f}，'
                f'精确上界为{summary["exact_mean"]:.6f}，192个查询中190个达到集合最优。此搜索同样提供给所有网络，收益分别报告。</p>')
    body.append('<h2>2. 为什么旧模型难以超过基线</h2><ol>'
                '<li><b>目标本身接近可直接求解。</b>当前只有19味候选，贪心每一步直接读取完整图谱计算精确边际增益；网络在学习近似同一个已知函数。新测试中贪心距集合最优仅约0.00135。</li>'
                '<li><b>训练示范和选模限制。</b>旧方案先模仿贪心，却不允许最终选择PPO前检查点。新增方案允许保留较好的初始模型，并另用仅训练查询的精确集合生成示范。</li>'
                '<li><b>折扣与终局评价不一致。</b>旧奖励是目标增量，却用0.97折扣；这会偏向较早的增益。新方案取gamma=1使累计增量等于最终目标，使用GAE、独立价值网络预热、较小学习率、衰减熵和KL提前停止。组合改动的结果不等于每一项都单独证明有效。</li>'
                '<li><b>样条实现差异。</b>旧正式网络使用高斯基函数。本轮新增三次B样条及隐藏层tanh限幅，更接近论文的样条边函数描述；样条本身并未在本轮取得优于MLP或贪心的证据。</li>'
                '<li><b>评价目标与论文不相同。</b>二值关联覆盖不是药物亲和度，也不会惩罚大量无关靶点。补算Jaccard后发现覆盖高不等于靶点选择性高。</li></ol>')
    body.append('<h2>3. 固定方案与新测试</h2><p>沿用464个训练查询，排除上一轮全部575个查询后，用预先固定的哈希顺序取96个验证、192个测试三症状组合。'
                '12个新模型（4方案×3种子）全部先在验证集冻结，再统一评估新测试。模型状态、语义指纹、代码指纹、验证轨迹和PPO诊断均保存。'
                '新测试仍与训练共用同一公开图谱，不是新的患者或未见知识。</p>'
                '<p>60轮示范学习、30轮价值预热、180轮PPO、每轮24个episode，三个固定种子42/43/44。'
                '最优集合示范只计算训练查询，测试最优解仅用于事后上界评价；动作掩码和参考目标在各方法间保持一致。</p>')
    rows = [[NAMES[k], f'{v["raw"]["objective"]:.6f}', f'{v["raw"]["objective_seed_std"]:.6f}',
             f'{v["shared_local_search"]["objective"]:.6f}', f'{v["raw"]["herb_count"]:.3f}',
             f'{v["shared_local_search"]["herb_count"]:.3f}'] for k,v in result.items()]
    headers = ["方案", "原始目标均值", "种子标准差", "统一搜索后目标", "原始候选数", "搜索后候选数"]
    body.append(table(headers,rows))
    with (OUT/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(10,4.8)); y=np.arange(len(result))
    ax.barh(y-.17,[v["raw"]["objective"] for v in result.values()],height=.3,label="原始策略",color="#5b7381")
    ax.barh(y+.17,[v["shared_local_search"]["objective"] for v in result.values()],height=.3,label="相同集合搜索后",color="#a8b4b9")
    ax.axvline(summary["exact_mean"],color="#9e613d",linestyle="--",label="精确集合最优")
    ax.set_yticks(y,[NAMES[k] for k in result]);ax.invert_yaxis();ax.set_xlim(.80,.83)
    ax.set_xlabel("参考综合目标（横轴放大为0.80—0.83）");ax.legend(loc="upper center",bbox_to_anchor=(.5,1.15),ncol=3,fontsize=9)
    body.append(chart(fig,"policy_comparison","图1：原始策略与统一后处理分开评价；横轴放大显示小幅差异。"))
    body.append(table(["方案", "三个种子选择的PPO迭代"],[[NAMES[k],str(v)] for k,v in summary["selected_iterations"].items()]))
    body.append('<p>配对比较先对每查询三个种子结果取平均，再按查询重采样；五组原始模型与贪心比较使用Holm校正。'
                '查询共享实体，独立性不成立，统计区间仅为给定图谱与查询集合下的描述。</p>')
    body.append(table(["原始方案减贪心", "均值差", "查询重采样95%区间", "Holm p"],[[NAMES[k],f'{v["mean_difference"]:+.6f}',str([round(x,6) for x in v["query_bootstrap_95ci"]]),f'{v["holm_p"]:.6f}'] for k,v in summary["raw_minus_greedy"].items()]))
    training=[json.loads(p.read_text(encoding="utf8")) for p in sorted((OUT/"training").glob('*.json'))]
    body.append('<h3>价值网络与训练审计</h3>'+table(["方案/种子","参数量","价值预热前MSE","预热后MSE","示范回退次数"],[[f'{NAMES[r["variant"]]} / {r["seed"]}',r["parameters"],f'{r["critic_mse_before"]:.5f}',f'{r["critic_mse_after"]:.5f}',r["teacher_fallbacks"]] for r in training]))
    body.append('<p>价值预热误差是训练示范状态上的拟合误差，不是未见病例误差。旧RBF隐藏层约35%的归一化输出位于基函数中心网格[-1,1]之外；高斯基仍有尾部响应，不能仅据此断言网络失效。新样条层前加入tanh，保证基函数的输入范围。</p>')
    body.append('<h2>4. 补齐论文形式的靶点一致性指标</h2><p>TCS按Jaccard计算：症状靶点与候选靶点交集除以并集。'
                '分别报告49个共享靶点计算范围，以及当前取得的所有显著关联靶点范围。两种范围都使用数据库关联，尚非论文预测的结合概率。</p>')
    body.append(table(["原始策略","共享49靶点Jaccard","全部显著关联靶点Jaccard"],[[NAMES[k],f'{v["raw"]["target_jaccard_shared49"]:.6f}',f'{v["raw"]["target_jaccard_all_eligible"]:.6f}'] for k,v in result.items()]))
    body.append('<p>大多数中药关联大量查询以外的靶点，导致全范围Jaccard明显低于覆盖率。不能通过只保留命中靶点缩小并集来靠近论文0.67。'
                '目前的差距指向化合物—靶点亲和度校准、数据对齐和目标选择性问题，而非仅增加训练轮数。</p>')
    body.append('<h2>5. STRING通路富集</h2><p>固定3个公开症状查询和3种策略（贪心、旧KAN、移除PPI的旧KAN，种子42），使用STRING 12.0官方API。'
                '主分析以49个可选靶点为背景，整个人类蛋白背景仅作敏感性分析。所有实际返回的KEGG/Reactome条目、请求参数及响应指纹均保留。'
                'API未返回条目不等于该通路不存在；显著性按来源返回的FDR解释，不宣称已跨查询组合统一校正。</p>')
    body.append(table(["查询","方法","背景","映射蛋白数","返回通路数","FDR<0.05条目数"],[["、".join(r["query"]),r["method"],r["background"],r["protein_count"],len(r["pathways"]),r.get("significant_pathways_fdr005","—")] for r in pathways["results"]]))
    body.append('<p>本轮同一查询的三种方法得到相同的靶点集合，因此富集结果相同。以可选靶点为背景未返回显著通路；以全人类为背景则返回较多。'
                '这不支持宣称“删除PPI必然使通路失去显著性”，也说明疾病相关数据库的预筛选不能被当作算法独立贡献。</p>')
    body.append('<h2>6. 外部200条专家标注记录：发现范围缺口</h2>'
                '<p>来源为TCMEval-SDT作者公开的200条带答案训练分区，本系统此前没有用它训练；输入仅使用Clinical Data，参考答案、推理和选项均不送入系统。'
                '作者测试/验证分区未公开完整答案，故本项称外部零样本审计，不冒称官方测试成绩。病例涉及广泛中医疾病，不等同于原论文糖尿病队列。</p>')
    body.append(table(["审计项","结果"],[["全部记录",external["records"]],["金标准证候标签总数",external["total_gold_labels"]],["按固定文字规则能映射当前图谱的标签",external["gold_labels_in_graph"]],["至少一个标签在图谱内的记录",external["records_with_any_gold_in_graph"]],["未识别阳性症状记录",external["records_with_no_positive_symptom"]],["输入显式提及金标准证候",external["explicit_gold_mentions"]],["重复临床文本",external["duplicate_text_excess"]]]))
    body.append(table(["方法","全200条首候选命中","前三命中","多标签micro-F1"],[[k,f'{v["all_records"]["top1_any_gold_hit"]:.2%}',f'{v["all_records"]["top3_any_gold_hit"]:.2%}',f'{v["all_records"]["multilabel_micro_f1"]:.6f}'] for k,v in external["evaluation"].items()]))
    body.append('<p>只有28/335个标签落入当前图谱，首候选命中1/200、前三命中2/200。去除输入中显式出现答案的2条后，结论未改变。'
                '单标签子集83条的首候选准确率为0，不能用大范围工程覆盖率代替辨证准确率。这里的精确文字匹配会受标签体系差异影响，尚无经专家审核的语义映射。</p>')
    body.append('<h2>7. 真实公开病例的监督辨证对照</h2><p>进一步取得TCM-SD作者公开数据，严格使用其训练/验证/测试分区。'
                '输入只包含主诉、病情描述和检查文本，不加入患者标识、疾病名称或证候标签字段；字符TF-IDF仅在训练文本拟合。'
                '比较ComplementNB与两种正则强度的SGD文本分类器，按去除训练重复文本后的验证macro-F1选型，模型冻结后才取得测试答案。</p>')
    body.append(table(["训练记录","验证记录","测试记录","训练标签数","词表大小"],[[supervised["train"],supervised["dev"],supervised["test"],148,supervised["vocabulary_size"]]]))
    body.append(table(["候选配置","验证macro-F1"],[[str(r["configuration"]),f'{r["unseen_text_validation"]["macro_f1_148_train_labels"]:.4f}'] for r in supervised["selection"]["candidates"]]))
    body.append('<p>选定配置：'+html.escape(str(supervised["selection"]["selected"]))+'。这些是监督文本基线，不是LLM或KAN/PPO辨证模型。</p>')
    body.append(table(["方法 / 范围","记录数","准确率","macro-F1（148类）","Kappa","前三命中"],[[f'{method} / {group}',v["records"],f'{v["accuracy"]:.4f}',f'{v["macro_f1_148_train_labels"]:.4f}',f'{v["kappa"]:.4f}',f'{v["top3_hit"]:.4f}'] for method,groups in supervised["evaluation"].items() for group,v in groups.items()]))
    body.append(table(["重复/标签审计","数量"],[["验证集与训练重复文本",supervised["validation_excluded_seen_texts"]],["测试与训练或验证重复文本",supervised["test_seen_texts_train_or_dev"]],["测试文本内部重复余量",supervised["duplicate_test_text_excess"]],["测试输入字面包含金标准证候",supervised["test_literal_gold_mentions"]],["测试行的源user_id在训练中出现",supervised["test_rows_with_user_id_in_train"]]]))
    body.append('<p>源user_id的具体语义尚待作者核实，不能自动等同患者ID；本报告不宣称完成了患者级独立划分。官方分区指标之外，另列去重文本结果与不包含字面答案的分层结果。'
                '原始记录与训练模型留在独立目录，不随源码发布；数据使用遵循作者标注的CC BY-NC-SA 4.0。</p>')
    if supervised["tcmeval_transfer"]:
        t=supervised["tcmeval_transfer"]
        body.append(f'<p>将冻结文本分类器直接迁移到TCMEval-SDT的200条记录：首候选命中{t["top1_any_gold_hit"]:.2%}、前三命中{t["top3_any_gold_hit"]:.2%}。'
                    '这属于探索性跨数据集检验，标签体系不一致，且此前已看过图谱基线结果；不作为新的未接触测试集。</p>')
    body.append('<h2>8. 论文对应状态与下一步</h2>'+table(["论文内容","本轮进展","尚缺的证据"],[
        ["KAN/PPO与消融","新增样条、稳定PPO、训练示范对照及全新查询留出","未证明网络优于同信息贪心；不能替代50例病例消融"],
        ["处方复杂度","统一局部集合搜索，190/192查询达到给定目标最优","无剂量、药效与临床处方适用性验证"],
        ["TCS/Jaccard","两种靶点范围的真实计算与逐查询结果","缺少经测量数据校准的亲和度模型"],
        ["PPI与通路富集","已执行官方富集并对照背景","当前数据不支持论文PPI消融显著性结论"],
        ["证候准确率/F1/Kappa","外部200条审计及TCM-SD官方分区监督对照","不是原论文86%/0.83的同队列复现"],
        ["LLM抽取和语义归一化","已有规则抽取与真实病例错误审计","尚未训练或验证论文的LLM抽取模块"],
        ["体质与禁忌","明确输入约束持续保留","缺少个体化疗效和完整临床规则标注"]]))
    body.append('<h2>9. 复现与来源</h2><pre>python scripts/research_v2.py --prepare\npython scripts/reproduce_policy_v2.py --legacy --seed 42\npython scripts/reproduce_policy_v2.py --legacy --seed 43\npython scripts/reproduce_policy_v2.py --legacy --seed 44\npython scripts/reproduce_policy_v2.py --seed 42\npython scripts/reproduce_policy_v2.py --seed 43\npython scripts/reproduce_policy_v2.py --seed 44\npython scripts/research_v2.py --aggregate\npython scripts/pathway_audit.py\npython scripts/external_syndrome_audit.py\npython scripts/tcm_sd_benchmark.py\npython scripts/verify_v2.py\npython scripts/report_v2.py</pre>'
                '<p>须先完成项目公开证据初始化。旧版对照权重由第一轮脚本生成。新模型选择与结果分别保存，原生产策略保持原版本；未根据新测试挑选生产种子。'
                '查询测试已被本轮分析，今后改动需新评估方案。不同计算环境或上游数据版本可能产生差异。</p><ul>'
                '<li><a href="https://arxiv.org/abs/2404.19756">KAN原始方法</a>；<a href="https://spinningup.openai.com/en/latest/algorithms/ppo.html">PPO与KL提前停止参考</a></li>'
                '<li><a href="https://string-db.org/help/api/">STRING官方富集接口与背景参数</a></li>'
                '<li><a href="https://doi.org/10.1038/s41597-025-04772-9">Wang等，TCMEval-SDT（2025）</a>；<a href="https://doi.org/10.6084/m9.figshare.27184596.v4">数据DOI</a></li>'
                '<li><a href="https://aclanthology.org/2022.ccl-1.80/">Ren等，TCM-SD（2022）</a>；<a href="https://github.com/Borororo/ZY-BERT">作者数据与许可</a></li></ul>')
    css='body{font:16px/1.8 "Microsoft YaHei",sans-serif;color:#25323a;margin:auto;max-width:1100px;padding:36px}h1{font-size:30px}h2{margin-top:40px;font-size:24px}header{border-bottom:2px solid #526874}aside{background:#f3f5f6;border-left:4px solid #526874;padding:18px;margin-top:24px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{border:1px solid #adb8be;padding:8px;text-align:left}th{background:#f3f5f6}figure{margin:24px 0}img{max-width:100%}figcaption{font-size:13px}pre{white-space:pre-wrap;background:#f3f5f6;padding:18px}@media print{body{font-size:11pt;padding:0}figure,tr{break-inside:avoid}h2{break-after:avoid}}'
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DMease第二轮研究报告</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>',encoding='utf8')
    print('Wrote experiments/v2/report.html and comparison.csv')


if __name__ == "__main__":
    main()

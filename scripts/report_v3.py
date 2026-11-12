"""Create the third-round standalone report from actual saved experiment outputs."""
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import build_experiment_report as render

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v3"


def read(name):
    return json.loads((OUT/f"{name}.json").read_text(encoding="utf8"))


def method(config):
    if config["method"] == "mean":
        return "训练均值"
    if config["method"] == "ridge":
        return f"岭回归 α={config['alpha']}"
    return f"随机森林 叶节点≥{config['min_samples_leaf']}"


def main():
    affinity, source, diagnostic, extraction = (read(name) for name in ("affinity", "affinity_sources", "syndrome_diagnostics", "extraction"))
    imbalance = read("imbalance")
    format_audit = read("extraction_format_audit")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "DMease-v3"})
    render.FIGURES = OUT/"figures"; render.FIGURES.mkdir(exist_ok=True)
    table, chart = render.table, render.chart
    body = ['<header><p>DMease · 第三轮可复现研究</p><h1>实测亲和度、语言模型抽取与跨数据集诊断</h1><p>公开数据 · 固定划分与提示词 · 保留全部结果</p></header>',
            '<aside>本轮推进化合物亲和度预测和真实语言模型调用。化合物实验、症状抽取和数据库推荐是不同层次的证据，'
            '不能互换为临床疗效。原论文50例糖尿病病例、医生配对结果及完整训练材料仍未取得，本报告不宣称复现原论文的数值。</aside>']
    body.append('<h2>1. 实测亲和度实验</h2><p>从ChEMBL官方接口获取人源PPARG、AKT1、INSR的精确Ki/Kd记录，'
                '仅保留高置信单蛋白结合实验。剔除潜在重复、异常质量标记及突变体，按标准化化合物—靶点—终点聚合重复测量。'
                '使用pK=9−log10(K/nM)，Ki与Kd分别建模。所有无环分子分到同一骨架组；骨架哈希固定划分，避免同一骨架跨训练、验证和测试。</p>')
    body.append(table(["靶点", "接口记录", "筛选后实测记录", "排除原因"], [
        [k, v["fetched"], v["eligible"], "; ".join(f"{reason}: {n}" for reason,n in v["excluded_first_reason"].items())] for k,v in source["audit"].items()]))
    body.append(table(["终点", "训练", "验证", "测试", "状态"], [[name, *[s["counts"][k] for k in ("train", "validation", "test")],
        "已评价" if s["status"] == "evaluated" else "未达到预设样本数"] for name,s in affinity["strata"].items()]))
    for name, stratum in affinity["strata"].items():
        if stratum["status"] != "evaluated":
            continue
        chosen = stratum["selection"]["selected_index"]
        test = stratum["test"][chosen]; baseline = stratum["test"][0]
        ci = stratum["comparison"]["ci95"]
        body.append(f'<h3>{html.escape(name)}：{stratum["counts"]["test"]}个测试化合物、{stratum["scaffolds"]["test"]}个测试骨架</h3>'
                    f'<p>验证集选出{method(stratum["selection"]["candidates"][chosen]["config"])}，测试MAE为{test["MAE_pK"]:.3f} pK，'
                    f'训练均值基线为{baseline["MAE_pK"]:.3f} pK。测试R²={test["R2"]:.3f}，Spearman={test["Spearman"]:.3f}。'
                    f'按测试骨架做2000次配对重采样，MAE差值的描述性95%区间为[{ci[0]:.3f}, {ci[1]:.3f}] pK。</p>')
        body.append(table(["模型", "验证MAE", "测试MAE", "测试RMSE", "测试R²", "验证选中"], [
            [method(c["config"]), f'{c["validation"]["MAE_pK"]:.4f}', f'{t["MAE_pK"]:.4f}', f'{t["RMSE_pK"]:.4f}', f'{t["R2"]:.4f}', "是" if i == chosen else ""]
            for i,(c,t) in enumerate(zip(stratum["selection"]["candidates"],stratum["test"]))]))
        fig, axes = plt.subplots(1,2,figsize=(10,4.1),layout="constrained")
        rows = stratum["predictions"]
        gold = [r["pK"] for r in rows]; prediction = [r["predictions"][chosen] for r in rows]
        axes[0].scatter(gold,prediction,color="#285f80",s=35,alpha=.8)
        low, high = min(gold+prediction)-.2, max(gold+prediction)+.2
        axes[0].plot([low,high],[low,high],"--",color="#555555",linewidth=1)
        axes[0].set(xlabel="实测 pKi", ylabel="预测 pKi", title="骨架留出测试")
        axes[1].bar(["训练均值","验证选中模型"],[baseline["MAE_pK"],test["MAE_pK"]],color=["#a4a4a4","#285f80"])
        axes[1].set(ylabel="平均绝对误差（pKi）", title="相同32个化合物")
        body.append(chart(fig,"affinity_scaffold","结构特征能够预测部分PPARG结合强度，但测试量有限，尚不能外推到整味中药。"))
        body.append(f'<p>训练与测试仍有{stratum["document_overlap_train_test"]}篇来源文献重叠，因此这是骨架隔离测试，'
                    '不是文献隔离验证。只纳入精确数值会产生选择偏差，实验条件差异也未完全消除。其他五个终点样本不足，未更换划分凑样本。</p>')
    body.append('<p>实现修复记录：首次运行时，RDKit去电荷操作后未重建环信息，误排除了8条PPARG记录。'
                '修复后沿用全部筛选条件、骨架哈希、模型候选与验证选模规则重算；修复前的结果与代码完整保存在initial_affinity_implementation。'
                '由于先前已观察过测试结果，修复后的数值属于更正评价，不算第二次独立验证。</p>'
                '<p>当前本地TCMSP成分表缺少可核验的结构标识连接，尚未把化合物预测强行映射到中药。'
                '输出是连续pK，不是校准后的结合概率；没有把关联显著性或STRING分数当作实测亲和度。</p>')
    body.append('<h2>2. 本地语言模型症状抽取</h2><p>固定版本Qwen2.5-1.5B-Instruct对TCMEval-SDT的200条公开记录逐条推理。'
                '输入只有Clinical Data；提示词、生成参数在计分前固定。每条仅生成一次，要求JSON结构和逐字原文证据。'
                '引用无法核验或结构不合法的输出被拒绝，失败记录仍参与计分。</p>'
                '<p>主要评价对齐系统24种标准症状。专家Clinical Information通过固定词典投影成参考标签，'
                '这是不完整的投影参考，不能当作完整临床标注准确率；没有参考症状的记录仍保留在总分母中。'
                '<b>参考投影和规则对照使用同一词典解析器，评价会偏向规则方法，不能作为公平的通用医学抽取排名。</b>'
                '这些指标主要衡量当前模型、提示词、输出协议与固定症状接口的兼容情况，也不能与论文的诊断F1直接比较。</p>')
    names = {"rules": "现有规则", "local_llm": "本地语言模型"}
    body.append(table(["抽取方法", "记录数", "微平均精确率", "微平均召回率", "微平均F1", "集合完全匹配"], [
        [names[m], s["canonical_all"]["records"], *[f'{s["canonical_all"][k]:.3f}' for k in ("precision","recall","f1","exact_set_rate")]] for m,s in extraction["summary"].items()]))
    rules_f1=extraction["summary"]["rules"]["canonical_all"]["f1"]
    llm_f1=extraction["summary"]["local_llm"]["canonical_all"]["f1"]
    if llm_f1 <= rules_f1:
        body.append('<p><b>当前本地模型、提示词与协议组合在严格投影评价中没有超过现有规则。</b>实际完成模型调用不等于验证有效；'
                    '格式稳定性和医学概念映射仍需改进，当前结果不支持直接替换现有症状解析流程。</p>')
    else:
        body.append('<p>本地模型在本次投影参考下的F1较高，但标签不完整和公开数据重用限制仍存在，需独立人工标注再验证。</p>')
    quality = extraction["quality"]
    body.append(f'<p>JSON格式失败{quality["invalid_json"]}条，达到输出长度上限{quality["output_limit"]}条，'
                f'输入截断{quality["input_truncated"]}条，投影参考为空{quality["empty_projected_reference"]}条。'
                f'单条生成耗时中位数{quality["median_seconds"]:.1f}秒。未通过逐字引用检查的条目数为{quality["rejected_items"].get("ungrounded",0)}。</p>')
    body.append(table(["有效JSON内的条目拒绝原因", "条目数"], list(quality["rejected_items"].items())))
    fig, ax = plt.subplots(figsize=(7.5,4),layout="constrained")
    xs = np.arange(3)
    for i,(m,s) in enumerate(extraction["summary"].items()):
        ax.bar(xs+(i-.5)*.34,[s["canonical_all"][k] for k in ("precision","recall","f1")],width=.34,label=names[m])
    ax.set_xticks(xs,["精确率","召回率","F1"]); ax.set_ylim(0,1); ax.legend(frameon=False)
    ax.set_title("24症状接口投影评价 · 全部200条记录\n参考由规则词典生成，评价偏向规则方法",fontsize=11)
    body.append(chart(fig,"local_extraction","同一词典生成参考，存在对规则有利的偏差；此图不能作为独立临床标注下的医学能力比较。"))
    body.append(table(["方法", "非空投影参考F1", "严格表面词F1（辅助诊断）", "下游证候首候选", "下游前三"], [
        [names[m], f'{s["canonical_nonempty_reference"]["f1"]:.3f}', f'{s["strict_surface_diagnostic"]["f1"]:.3f}',
         f'{s["downstream_exploratory"]["top1"]:.1%}', f'{s["downstream_exploratory"]["top3"]:.1%}'] for m,s in extraction["summary"].items()]))
    body.append('<p>表面词比较要求与专家短语逐字一致，不处理临床同义关系，只作误差诊断。'
                '这些公开记录在上一轮已用于外部辨证审计，不是全新官方测试集；也无法排除模型预训练见过公开内容。'
                '抽取结果尚需人工核对；此次研究未自动替换正式辅助分析流程。</p>')
    body.append('<h3>格式适配的事后对照</h3><p>观察到格式问题后，单独增加确定性适配：仅移除完整JSON对象后的一个多余引号，'
                '或完整有效JSON外层的Markdown代码框。保留原始生成与严格评分，不补括号、不补字段、不补截断文本，也不丢弃任意附加文字。'
                '此项复用相同输出，属于事后工程分析。</p>')
    fs=format_audit["summary"]
    body.append(table(["评价", "JSON失败数", "精确率", "召回率", "F1"], [
        ["原始严格规则",quality["invalid_json"],*[f'{extraction["summary"]["local_llm"]["canonical_all"][k]:.3f}' for k in ("precision","recall","f1")]],
        ["格式适配后（事后对照）",format_audit["remaining_json_failures"],*[f'{fs[k]:.3f}' for k in ("precision","recall","f1")]]]))
    body.append(table(["处理类别", "记录数"],list(format_audit["format_adjustments"].items())))
    body.append('<h2>3. 跨数据集与少样本证候问题</h2><p>复用上一轮冻结的TCM-SD文本分类器，不重新训练。'
                '以下属于观察结果后的描述性分析，用来定位问题，不构成新的独立验证。</p>')
    transfer = diagnostic["transfer"]
    body.append(f'<p>TCMEval的335个专家证候标签中，只有{transfer["gold_labels_in_training_vocabulary"]}个落在分类器标签表内。'
                f'200条病例中仅{transfer["records_with_any_known_gold"]}条有至少一个已知标签，固定标签表下首候选命中率的理论上限为'
                f'{transfer["top1_known_label_ceiling"]:.1%}；实际总体命中{transfer["overall_top1"]:.1%}，在有已知标签的子集内命中'
                f'{transfer["conditional_top1_known_gold"]:.1%}。因此标签缺口与已覆盖类别中的识别误差同时存在。</p>')
    body.append(table(["训练样本/证候", "证候数", "测试记录", "准确率", "前三命中率", "宏平均F1"], [
        [r["training_frequency"],r["labels"],r["test_records"], *[f'{r[k]:.3f}' if r[k] is not None else "—" for k in ("top1_accuracy","top3_hit","macro_f1")]] for r in diagnostic["frequency_bands"]]))
    body.append(table(["数据", "记录", "文本长度中位数", "字符2/3元特征覆盖均值"], [
        [name,r["records"],r["median_chars"],f'{r["mean_ngram_in_vocabulary"]:.1%}'] for name,r in diagnostic["feature_coverage"].items()]))
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
    bands=diagnostic["frequency_bands"]
    axes[0].bar([r["training_frequency"] for r in bands],[r["top1_accuracy"] for r in bands],color="#285f80")
    axes[0].set(xlabel="每类训练样本数",ylabel="同源测试准确率",ylim=(0,1),title="少样本类别明显落后")
    cov=diagnostic["feature_coverage"]
    axes[1].bar(["训练","同源测试","外部病例"],[r["mean_ngram_in_vocabulary"] for r in cov.values()],color=["#8b8b8b","#285f80","#bb683c"])
    axes[1].set(ylabel="字符特征覆盖率",ylim=(0,1),title="外部病例写法差异")
    body.append(chart(fig,"transfer_diagnostics","仅定位领域差异与类别不均衡，未根据测试结果重选模型。"))
    body.append('<h2>4. 类别加权改进实验</h2><p>预先固定6种样本加权和正则强度组合，复用只在原训练集拟合的文本特征。'
                '权重仅由训练类别频次计算，并归一化为均值1。与冻结的原基线一起，根据不重复验证文本上的148类宏平均F1选择模型，'
                '在新的测试计分前保存选模记录。该实验在观察原测试表现后开展，复用了公开测试集，属于探索性改进。</p>')
    selection = imbalance["selection"]
    body.append(table(["候选", "配置", "验证准确率", "验证宏平均F1", "验证选中"], [
        [i, json.dumps(c["configuration"],ensure_ascii=False),f'{c["validation"]["accuracy"]:.3f}',
         f'{c["validation"]["macro_f1_148_train_labels"]:.3f}', "是" if i == selection["selected_index"] else ""]
        for i,c in enumerate(selection["candidates"])]))
    balance_names={"frozen_unweighted_v2":"原未加权基线", "validation_selected":"验证选中方案"}
    body.append(table(["方法", "范围", "记录数", "准确率", "宏平均F1", "Kappa", "前三命中率"], [
        [balance_names[m], "完整测试" if group == "official_test" else "排除重复文本", r[group]["records"],
         *[f'{r[group][k]:.3f}' for k in ("accuracy","macro_f1_148_train_labels","kappa","top3_hit")]]
        for m,r in imbalance["results"].items() for group in ("official_test","unseen_text_test")]))
    body.append(table(["方法", "每类训练样本数", "测试记录数", "准确率", "宏平均F1"], [
        [balance_names[m], b["training_frequency"], b["test_records"],f'{b["accuracy"]:.3f}',f'{b["macro_f1"]:.3f}']
        for m,r in imbalance["results"].items() for b in r["frequency_bands"]]))
    fig,ax=plt.subplots(figsize=(8,4.3),layout="constrained")
    xs=np.arange(4)
    for i,(m,r) in enumerate(imbalance["results"].items()):
        ax.bar(xs+(i-.5)*.35,[b["accuracy"] for b in r["frequency_bands"]],width=.35,label=balance_names[m])
    ax.set_xticks(xs,[b["training_frequency"] for b in r["frequency_bands"]]);ax.set_ylim(0,1)
    ax.set(xlabel="每类训练样本数",ylabel="测试准确率",title="类别加权：按样本量观察实际收益与代价")
    ax.legend(frameon=False)
    body.append(chart(fig,"class_weighting","选模仅使用验证集；测试集已在上一轮使用，结果属于后续探索。"))
    body.append(table(["方法", "外部记录数", "外部首候选", "外部前三"], [
        [balance_names[m],r["external_transfer"]["records"], f'{r["external_transfer"]["top1"]:.1%}',f'{r["external_transfer"]["top3"]:.1%}']
        for m,r in imbalance["results"].items()]))
    body.append('<p>即使同源测试改善，跨数据集仍受标签体系和文本差异限制。类别加权结果不归为KAN/PPO收益，'
                '也未自动替换正式诊疗流程。未使用病例标识或证候答案作为特征。</p>')
    before=imbalance["results"]["frozen_unweighted_v2"]; after=imbalance["results"]["validation_selected"]
    body.append(f'<p>选中方案的外部首候选命中率为{after["external_transfer"]["top1"]:.1%}，原基线为{before["external_transfer"]["top1"]:.1%}。'
                f'高频类别（每类训练≥1000条）准确率从{before["frequency_bands"][-1]["accuracy"]:.1%}变为'
                f'{after["frequency_bands"][-1]["accuracy"]:.1%}。这两项代价与少样本收益一起报告。</p>')
    body.append('<h2>5. 对论文实现的影响</h2><ul><li>已新增实测Ki/Kd数据处理、结构指纹、骨架划分、模型训练和误差统计。'
                '连续亲和度预测不等于整个中药的结合概率，还需要可靠的成分结构连接和独立外部验证。</li>'
                '<li>已新增可实际运行的本地语言模型抽取、引用核验和固定200条记录的对照实验。'
                '词表、同义词与证候标签体系仍是关键限制，需要经核验的映射与独立标注集。</li>'
                '<li>KAN/PPO上一轮仍未超过贪心基线。本轮没有改变原测试集、奖励或原结果来使结论变好。'
                '只有在亲和度与新证据连接可靠后，才适合另行冻结训练和测试方案检验新奖励。</li>'
                '<li>保留抽取失败、样本不足的终点、修复前计算和负面结果。未用构造病例补充真实研究的分母。</li></ul>')
    body.append('<h2>6. 来源与复现</h2><p><a href="https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services">ChEMBL官方接口</a> · '
                '<a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct">Qwen模型及许可证</a> · '
                '<a href="https://doi.org/10.1038/s41597-025-04772-9">TCMEval数据论文</a> · '
                '<a href="https://github.com/Borororo/ZY-BERT">TCM-SD作者仓库</a></p>'
                '<p>protocol.json固定方案；affinity_sources.json保存API请求与校验值；各selection文件记录测试前选模；'
                'affinity.json、extraction.json、syndrome_diagnostics.json保存可复算数值。'
                '公开文件不含病例原文、模型生成文本或第三方大模型权重。运行scripts/verify_v3.py可离线核验来源与数值一致性。</p>')
    css = "body{font:16px/1.7 'Microsoft YaHei',sans-serif;max-width:1100px;margin:auto;padding:35px;color:#222}header{border-bottom:3px solid #333}h2{margin-top:40px}aside{padding:16px;background:#f2f2f2;border-left:4px solid #555}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #bbb;padding:8px;text-align:left}th{background:#eee}figure{margin:25px 0}img{max-width:100%}figcaption{color:#555;font-size:14px}a{color:#17597b}"
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DMease 第三轮实验报告</title><style>'+css+'</style><body>'+"\n".join(body)+'</body></html>'
    (OUT/"report.html").write_text(document, encoding="utf8")
    print(OUT/"report.html")


if __name__ == "__main__":
    main()

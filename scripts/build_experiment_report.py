"""Build a self-contained, printable report from published experiment results."""
from pathlib import Path
import base64
import csv
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
NAMES = {"full": "KAN + PPO", "mlp": "MLP + PPO", "without_ppi": "移除 PPI",
         "without_direct": "移除直接关系", "without_targets": "移除靶点及 PPI",
         "warm_start_only": "仅模仿预训练", "ppo_only": "仅 PPO"}


def read(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf8"))


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def chart(fig, name, caption):
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    svg = FIGURES / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines()) + "\n", encoding="utf8", newline="\n")
    plt.close(fig)
    encoded = base64.b64encode((FIGURES / f"{name}.png").read_bytes()).decode()
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def main():
    FIGURES.mkdir(exist_ok=True)
    for font in (Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "svg.hashsalt": "DMease-experiments-v1"})
    summary, optimal = read("summary"), read("optimality")
    before, after = read("parser_before"), read("parser_after")
    sensitivity, missing, constraints = (read(k)["results"] for k in ("sensitivity", "missing_evidence", "constraints"))
    baseline = summary["baseline_greedy"]
    variants = summary["ablations"]
    body = ['<header><p class="eyebrow">DMease · 公开证据实验记录</p><h1>模型、证据与工程可靠性评价</h1>'
            '<p>固定方案 · 三个随机种子 · 逐查询结果可追溯</p></header>']
    body.append('<div class="notice">本报告使用官方数据库关联与症状检索组合，不包含患者病例或专家诊疗标签。'
                '综合目标衡量给定图谱上的证据覆盖与候选数量成本，不代表诊断准确率、药效或临床安全性。</div>')
    body.append('<h2>1. 主要发现</h2><ul>'
                f'<li>完整 KAN/PPO 平均目标为 {variants["full"]["seed_mean"]["objective"]:.6f}，证据增益基线为 {baseline["mean"]["objective"]:.6f}。当前结果不支持复杂模型优于简单基线。</li>'
                '<li>移除直接关系或靶点通道明显降低同一参考目标；KAN 与 MLP 的差异、PPI 的独立增量未得到本实验充分支持。</li>'
                '<li>仅模仿预训练优于继续进行 PPO 的完整模型，提示当前奖励、任务规模和优化过程仍需改进；不把 PPO 的加入自动视为收益。</li>'
                f'<li>{summary["constraints_evaluated"]:,} 次约束查询执行中，指定规则违规为 {summary["constraint_violations"]}；精确枚举发现 7/62 个查询仍有优于贪心的集合。</li>'
                '<li>修复了文本否定范围、当前状态选择和包含关系的重复识别；开发用例通过不等于独立文本评价。</li></ul>')
    body.append('<h2>2. 数据、划分与实验控制</h2><p>当前计算范围为24个症状、19味中药、49个共享显著靶点和52条范围内STRING关系。'
                '训练/验证/测试分别为464/49/62个完整症状查询组合，底层图谱共享。它们覆盖单项、双项和确定性抽样的三项查询，不能反映真实病例分布。</p>'
                '<p>随机种子42、43、44；每个变体使用相同查询划分。默认60轮模仿预训练、120轮PPO、每轮12个episode、每10轮验证。'
                '模型依据完整参考目标的验证集表现选择检查点；仅模仿变体固定取第60轮。完整模型不允许选择PPO前检查点，便于单独检验后续PPO阶段。'
                '所有最终候选集合均在未删除关系的完整参考图谱、默认成本0.02、PPI权重0.05下评分，避免比较不同奖励定义。</p>'
                '<p>KAN参数111,956，MLP参数106,100（少约5.2%），为近似参数预算对照，不是参数完全相等的实验。'
                '“移除靶点”同时移除其依赖的PPI通道；各变体保持状态维数一致。方案在本轮主要运行前固定，未在外部平台预注册。'
                '不根据这次测试结果替换生产模型或调整默认参数。</p>')
    body.append('<h2>3. 模型与消融</h2>')
    rows = [[NAMES[k], f'{v["seed_mean"]["objective"]:.6f}', f'{v["seed_std"]["objective"]:.6f}',
             f'{v["seed_mean"]["symptom_evidence_coverage"]:.4f}', f'{v["seed_mean"]["herb_count"]:.3f}', v["parameters"]]
            for k, v in variants.items()]
    rows.append(["证据增益（确定性）", f'{baseline["mean"]["objective"]:.6f}', "—", f'{baseline["mean"]["symptom_evidence_coverage"]:.4f}', f'{baseline["mean"]["herb_count"]:.3f}', "—"])
    headers = ["方法", "综合目标均值", "种子间样本标准差", "症状证据覆盖", "平均候选数", "参数量"]
    body.append(table(headers, rows))
    with (OUT / "ablation_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    order = list(variants)
    ax.barh([NAMES[k] for k in order], [variants[k]["seed_mean"]["objective"] for k in order],
            xerr=[variants[k]["seed_std"]["objective"] for k in order], color="#546c79", capsize=3)
    ax.axvline(baseline["mean"]["objective"], color="#ac613b", linestyle="--", label="证据增益基线")
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("完整参考图谱上的综合目标（非临床指标）"); ax.legend(loc="lower right")
    body.append(chart(fig, "ablation", "图1：三种子均值；误差线为种子间样本标准差，不是置信区间。横轴从0起。"))
    body.append('<h3>配对差值</h3><p>先对同一查询的三个种子差值取平均，再按62个查询重采样10,000次。'
                '符号翻转20,000次，六组完整模型比较使用Holm校正。查询组合共享症状与图谱，独立性未建立，因此以下区间与p值仅作条件性描述；不推断临床总体。</p>')
    body.append(table(["完整KAN/PPO减去对照", "均值差", "查询重采样95%区间", "Holm调整p", "胜 / 平 / 负"], [
        [NAMES[k], f'{v["mean_difference"]:+.6f}', ' / '.join(f'{x:+.6f}' for x in v["query_bootstrap_95ci"]),
         f'{v["holm_p"]:.6f}', ' / '.join(map(str, v["wins_ties_losses"]))] for k, v in summary["full_minus_variant"].items()]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key in ("full", "mlp", "ppo_only"):
        histories = [read(f"{key}_{s}")["validation_history"] for s in (42, 43, 44)]
        x = [r["iteration"] for r in histories[0]]
        y = np.array([[r["reference_objective"] for r in h] for h in histories])
        axes[0].plot(x, y.mean(axis=0), marker=".", label=NAMES[key])
        axes[0].fill_between(x, y.mean(axis=0)-y.std(axis=0, ddof=1), y.mean(axis=0)+y.std(axis=0, ddof=1), alpha=.12)
    axes[0].set(xlabel="PPO迭代", ylabel="验证集参考目标", ylim=(0, 1)); axes[0].legend(fontsize=8)
    axes[1].bar(["全部查询", "靶点证据可用", "PPI证据可用"], [62, baseline["target_available_queries"], baseline["ppi_available_queries"]], color="#546c79")
    axes[1].set(ylabel="测试查询数", ylim=(0, 70)); axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    body.append(chart(fig, "training_and_availability", "图2：左为验证轨迹，阴影为种子标准差；右为证据可用范围。"))
    body.append(f'<p>靶点证据仅在53/62个查询中可用；基线在这些查询上的靶点覆盖均值为{baseline["target_coverage_when_available"]:.4f}。'
                'PPI证据在41/62个查询中可用，其条件覆盖为1.0000。全体查询的PPI覆盖0.6613，主要反映缺失比例，不能据此声称网络覆盖较差。'
                '基线症状覆盖已达到忽略数量和配伍限制的可用关联上界均值0.8683；未覆盖部分优先指向当前数据范围的缺口。</p>')
    body.append('<h2>4. 参数敏感性与证据缺失</h2><p>14组参数设置采用确定性证据增益策略；每次仅改变一个参数。'
                '评价仍使用默认参考目标。结果属于测试集描述，不用于调参。模型缺失证据测试固定完整图谱训练后的权重，仅在推理时删除关系。</p>')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, parameter, label in zip(axes, ("herb_cost", "max_steps", "ppi_weight"), ("候选成本", "最大候选数", "PPI选择权重")):
        rs = [r for r in sensitivity if r["parameter"] == parameter]
        ax.plot([r["value"] for r in rs], [r["mean"]["objective"] for r in rs], "o-", color="#546c79")
        ax.set(xlabel=label, ylim=(.45, .86)); ax.grid(alpha=.15)
    axes[0].set_ylabel("默认参考目标"); fig.tight_layout()
    body.append(chart(fig, "sensitivity", "图3：单参数变化；纵轴为同一参考目标，范围0.45—0.86。"))
    body.append(table(["参数", "取值", "参考目标", "平均候选数"], [[r["parameter"], r["value"], f'{r["mean"]["objective"]:.6f}', f'{r["mean"]["herb_count"]:.3f}'] for r in sensitivity]))
    missing_rows = []
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    for mode, label in (("greedy", "证据增益"), ("model", "固定KAN/PPO")):
        means, stds = [], []
        for fraction in (.1, .25, .5):
            scores = [r["mean"]["objective"] for r in missing if r["mode"] == mode and r["drop_fraction"] == fraction]
            means.append(np.mean(scores)); stds.append(np.std(scores, ddof=1))
            missing_rows.append([label, f"{fraction:.0%}", f"{means[-1]:.6f}", f"{stds[-1]:.6f}"])
        ax.errorbar([10, 25, 50], means, yerr=stds, marker="o", capsize=4, label=label)
    ax.set(xlabel="移除关系比例（%）", ylabel="完整参考图谱上的综合目标", ylim=(.5, .9)); ax.legend(); ax.grid(alpha=.15)
    body.append(chart(fig, "missing_evidence", "图4：证据缺失下的参考目标；误差线为三个运行的样本标准差。"))
    body.append(table(["策略", "移除比例", "平均目标", "标准差"], missing_rows))
    body.append('<p>按固定种子分别从非零直接关系、症状—靶点和中药—靶点矩阵中随机删去对应比例的条目；不新增关系，PPI边本身不直接抽样删除。'
                '小矩阵因取整可能偏离名义比例。三个种子同时决定训练随机性和删除位置，未做模型种子×删除种子的全交叉，不能分解两种方差来源。'
                '固定模型在本轮删除试验中均略高于对应贪心，但它曾在完整证据上训练；这不是未见知识泛化或普遍鲁棒性优势的证明。</p>')
    body.append('<h2>5. 约束与精确最优解审计</h2>')
    labels = {"exclude_first_baseline_choice": "排除基线首选", "exclude_all": "全部排除", "only_one_allowed": "仅允许一种", "forbid_first_two_as_pair": "首两味不得同时入选"}
    body.append(table(["场景", "策略", "执行查询次数（3种子）", "其中规则有效次数", "指定规则违规"], [
        [labels[scenario], mode, sum(r["queries"] for r in constraints if r["scenario"] == scenario and r["mode"] == mode),
         sum(r["active_constraint_queries"] for r in constraints if r["scenario"] == scenario and r["mode"] == mode),
         sum(r["violation_total"] for r in constraints if r["scenario"] == scenario and r["mode"] == mode)]
        for scenario in labels for mode in ("greedy", "model")]))
    body.append('<p>1,488次是同一62查询在场景、策略和种子下的重复执行次数，不是1,488个独立病例。'
                '基线候选不足两味时配伍场景没有有效规则，表中单列规则有效次数。规则是测试输入，不能解释为医学禁忌。妊娠停止、账号权限等由系统回归测试另行覆盖。</p>')
    stats = optimal["exact_minus_greedy"]
    body.append(f'<p>穷举检查{summary["exact_subsets_evaluated"]:,}个可行子集（含空集，每查询最多8味），允许考察单步暂时没有正增益的集合。'
                f'精确集合最优目标比贪心平均高{stats["mean_difference"]:.6f}，7个查询更高、55个相同。'
                '这说明贪心仍存在小范围优化余地。最优仅针对给定关联和成本函数，不是最佳处方；枚举的计算量随候选规模指数增长。</p>')
    query_file = OUT / "queries.json"
    queries = json.loads(query_file.read_text(encoding="utf8"))["splits"] if query_file.exists() else {}
    lookup = {q["id"]: "、".join(q["symptoms"]) for q in queries.get("test", [])}
    bmap = {r["query_id"]: r for r in optimal["greedy"]["rows"]}
    gaps = sorted([(r, bmap[r["query_id"]]) for r in optimal["exact"] if r["objective"]-bmap[r["query_id"]]["objective"] > 1e-6], key=lambda pair: pair[0]["objective"]-pair[1]["objective"], reverse=True)
    body.append(table(["公开症状查询", "贪心目标", "精确目标", "差值"], [[lookup.get(a["query_id"], a["query_id"][:12]), f'{b["objective"]:.6f}', f'{a["objective"]:.6f}', f'{a["objective"]-b["objective"]:.6f}'] for a, b in gaps]))
    body.append('<h2>6. 症状解析开发回归</h2><p>20条人工编写语言用例与24个症状×6种语法模板，共164条。'
                '用例由开发过程编写，参与修复；无专家标注、无病例文本、无独立留出验证。实体与存在状态同时一致才算匹配。</p>')
    body.append(table(["版本", "范围", "完全一致 / 总数", "实体状态micro-F1"], [[label, group, f'{v["exact_matches"]}/{v["cases"]}', f'{v["entity_state_micro_f1"]:.4f}'] for label, report in (("修复前", before), ("修复后", after)) for group, v in report["summary"].items()]))
    body.append('<p>修复涉及“无明显诱因出现口渴”等非症状否定、“并非没有”的双重否定、“已消失”的后置否定、'
                '“目前/今天”等当前状态标记，以及“夜尿频多”包含“尿频”时的重复识别。保留了另处单独提到“尿频”的独立匹配。'
                '复杂家族史、他人症状、假设语句、不确定表达与复杂时间关系仍需扩展评价；规则无法替代论文所述大模型抽取验证。</p>')
    body.append('<h2>7. 复现与后续实验边界</h2><p>results目录保留21份训练记录、逐查询评分、验证轨迹、模型指纹、14组参数、18组缺失关系和24组约束结果。'
                'protocol.json固定方案，queries.json记录查询划分，parser_cases.json固定文本开发用例。原始数据库与试验权重独立存放；结果不包含患者资料。</p>'
                '<pre>python scripts/run_experiments.py --seed 42\npython scripts/run_experiments.py --seed 43\npython scripts/run_experiments.py --seed 44\npython scripts/run_experiments.py --aggregate\npython scripts/evaluate_parser.py --label after\npython scripts/verify_experiment_artifacts.py\npython scripts/build_experiment_report.py</pre>'
                '<p>训练需先按项目README取得与方案指纹一致的公开证据包。网站数据更新后不能声称与旧快照相同；报告构建只需已发布的JSON结果和matplotlib。'
                '修复前结果对应基础提交4826e24中的解析器，在同一固定用例和证据包上运行所得，不能用当前解析器重跑并覆盖before结果。</p>'
                '<p>尚未完成的论文级实验包括：独立病例与专家评价、亲和度模型、完整实体对齐、体质适配证据、大模型抽取及独立安全规则验证。'
                '后续方法改动应使用新的冻结评估方案与独立测试材料；本轮测试集已被分析，不能继续作为未接触的最终测试集。</p>')
    body.append('<details><summary>运行环境与来源指纹</summary><pre>' + html.escape(json.dumps(summary["provenance"], ensure_ascii=False, indent=2)) + '</pre></details>')
    body.append('<footer>完成时间：' + html.escape(summary["completed_at"]) + ' · DMease研究记录 · 非临床评价</footer>')
    css = 'body{font:16px/1.8 "Microsoft YaHei",sans-serif;color:#25323a;margin:0 auto;padding:36px;max-width:1080px}header{border-bottom:2px solid #344b56;padding-bottom:20px}h1{font-size:30px;margin:8px 0}h2{font-size:23px;margin-top:42px}h3{font-size:18px}.eyebrow,footer{font-size:13px;color:#63737c}.notice{border-left:4px solid #63737c;background:#f4f6f7;padding:16px;margin:24px 0}table{width:100%;border-collapse:collapse;font-size:13px;margin:18px 0}th,td{border:1px solid #aab4ba;padding:8px;text-align:left}th{background:#f4f6f7}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#57656e}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f7;padding:16px;font-size:12px}footer{border-top:1px solid #aab4ba;margin-top:36px;padding-top:16px}@media print{body{padding:0;font-size:11pt}table{font-size:9pt}h2{break-after:avoid}figure,tr{break-inside:avoid}details{display:none}}'
    (OUT / "report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DMease公开证据实验报告</title><style>' + css + '</style><body>' + ''.join(body) + '</body></html>', encoding="utf8")
    print("Wrote experiments/report.html, ablation_summary.csv and four figure pairs")


if __name__ == "__main__":
    main()

"""Standalone research report and publication-friendly static figures."""
import base64
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from chemistry_v5 import ROOT,OUT,read,save
from dmease.molecular_feature_graph import PlanningFeatureGraph
from dmease.molecular_planning import MolecularEvidence

FIG=OUT/'figures'


def clean_svgs():
    for path in FIG.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'figure.dpi':150,'savefig.bbox':'tight'})
    graph=PlanningFeatureGraph(OUT/'planning_features.json');chem=MolecularEvidence(OUT/'molecular_bundle.json',graph)
    audit=chem.data['audit'];identity=read(OUT/'identity_audit.json');affinity=read(OUT/'affinity_evaluation.json')
    fig,ax=plt.subplots(figsize=(12,4.3));ax.set_xlim(0,12);ax.set_ylim(0,4.3);ax.axis('off')
    blocks=[(.2,2.5,'药材与成分\nTCMSP / SymMap'),(3.2,2.5,'身份检查与结构\nPubChem + RDKit'),(6.2,2.5,'实测 Ki / Kd\nChEMBL 来源可追溯'),(9.2,2.5,'结构指纹回归\n来源组留出 / 适用域'),
            (.2,.5,'症状与显式约束\n独立的研究靶点上下文'),(3.2,.5,'状态 / 动作特征\n已覆盖分子索引 + 缺失掩码'),(6.2,.5,'KAN 策略 + PPO\n逐步重算收益 / 显式停止'),(9.2,.5,'候选与逐步解释\n来源、预测误差、保存导出')]
    for x,y,label in blocks:
        ax.add_patch(FancyBboxPatch((x,y),2.5,1.1,boxstyle='round,pad=.1',edgecolor='#365d73',facecolor='#edf3f6'))
        ax.text(x+1.25,y+.55,label,ha='center',va='center',fontsize=9)
    for y in (3.05,1.05):
        for x in (2.8,5.8,8.8):ax.annotate('',(x+.25,y),(x-.05,y),arrowprops={'arrowstyle':'->','color':'#365d73'})
    ax.annotate('',(4.45,1.75),(7.45,2.35),arrowprops={'arrowstyle':'->','color':'#365d73','connectionstyle':'angle3'})
    ax.annotate('',(7.45,1.75),(10.45,2.35),arrowprops={'arrowstyle':'->','color':'#365d73','connectionstyle':'angle3'})
    ax.set_title('已接通的分子证据与序贯推荐路径',fontweight='bold',pad=14)
    fig.savefig(FIG/'architecture.png');fig.savefig(FIG/'architecture.svg');plt.close(fig)
    values,known,winners=chem.matrices();fig,ax=plt.subplots(figsize=(10,8))
    masked=np.ma.masked_where(~known,values);cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#dddddd')
    image=ax.imshow(masked,vmin=0,vmax=.4,cmap=cmap,aspect='auto')
    ax.set_xticks(range(len(chem.channels)),chem.channels,rotation=30,ha='right');ax.set_yticks(range(len(graph.herbs)),[h['name'] for h in graph.herbs])
    for (i,j),r in winners.items():ax.text(j,i,('M' if r['affinity']['kind']=='measured' else 'P')+f' {values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>.2 else 'black')
    ax.set_title('药材—靶点/终点索引：M 实测；P 预测下界；灰色 缺失',pad=16)
    fig.colorbar(image,ax=ax,label='检索索引（非结合概率）',shrink=.65);fig.tight_layout();fig.savefig(FIG/'molecular_matrix.png');fig.savefig(FIG/'molecular_matrix.svg');plt.close(fig)
    if not (OUT/'integration_analysis.json').exists():
        clean_svgs()
        print('Architecture and measured molecular matrix rendered; planning evaluation still pending.',flush=True)
        return
    analysis=read(OUT/'integration_analysis.json');planning=read(OUT/'planning_summary.json')
    labels=['贪心：不计分子证据','贪心：仅实测','贪心：实测＋域内预测','完整训练KAN/PPO','KAN/PPO＋贪心核对']
    keys=['without_molecular','measured_only','molecular_greedy','final_ppo','final_ppo_guarded']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,field,title in zip(axes,['objective','molecular_index'],['统一完整目标下的得分','选择集合的分子证据索引']):
        vals=[analysis['summary'][k][field] for k in keys]
        bars=ax.barh(labels,vals,color=['#9aa4ae','#7d96a2','#4a768c','#33607c','#254252'])
        ax.invert_yaxis();ax.set_title(title);ax.set_xlim(0,max(vals)*1.18)
        for bar,v in zip(bars,vals):ax.text(v+.002,bar.get_y()+bar.get_height()/2,f'{v:.4f}',va='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(FIG/'planning_comparison.png');fig.savefig(FIG/'planning_comparison.svg');plt.close(fig)
    esc=lambda x:html.escape(str(x),quote=True)
    def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'
    def picture(name):return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((FIG/(name+'.png')).read_bytes()).decode()+'">'
    body='<h1>DMease 分子链路与推荐整合报告</h1><p>第五轮 · 公共化学证据、结构预测与 KAN/PPO 实际整合</p>'
    body+='<p>药材—成分—分子结构—靶点亲和度已连接到推荐状态、收益和界面；本轮验证的是研究检索过程，不是临床治疗有效性。</p>'+picture('architecture')
    body+='<h2>1. 数据身份与覆盖</h2>'+table(['检查项','实际数量'],[
        ['原药材—成分连接',1853],['唯一CID连接',1191],['结构获取成功的CID',867],
        ['质量核对后连接',audit['resolved_herb_compound_links']],['保留结构',audit['unique_compounds']],
        ['隔离冲突连接',len(identity['rejected'])],['覆盖药材/候选范围',f"{len(chem.data['herbs'])}/19"],
        ['成分实测Ki/Kd记录',audit['measured_compound_target_endpoint_records']],['域内预测记录',audit['predicted_compound_target_endpoint_records']],
        ['缺少实测或可用预测的成分/靶点/终点组合',audit['unknown_compound_target_endpoint_records']]])
    body+='<p>按显式TCMSP编号和PubChem CID连接，排除Suppressed、多个CID、缺号、分子量或可解析元素式冲突。麦冬缺少此来源的成分链路，未用相似药名补齐。最终成分结构保留立体信息，实测值只按相同标准化立体母体和完整InChIKey匹配。原始TCMSP立体结构尚未独立取得；质量检查不能排除所有来源错误。</p>'
    body+='<p>训练中发现身份冲突后中断初轮训练，在测试前统一过滤并重训，过程见identity_amendment.json。接口访问出现ChEMBL HTTP500/超时及SymMap关系接口502；因此本轮使用此前已核验的ChEMBL37实测快照，新增结构来自PubChem实际查询，不把旧测量称作新采集。</p>'
    body+='<h2>2. 实测和结构预测</h2>'+table(['通道','纳入记录','留出评价','来源组','模型MAE','均值基线MAE','误差90%分位','允许域内预测'],[
        [k,r['rows'],r['evaluated_rows'],r['evaluated_components'],round(r.get('model',{}).get('MAE_pK',0),4) if r.get('model') else '缺少评价',
         round(r.get('baseline',{}).get('MAE_pK',0),4) if r.get('baseline') else '缺少评价',round(r.get('absolute_error_q90',0),4) if r.get('model') else '—',r['deployable']] for k,r in affinity['strata'].items()])
    body+='<p>Ridge α=10、Morgan半径2/2048位，按共享文献、结构和骨架的连通组留出。超参数与数据已在前轮探索，结果不是独立外部确认。每折至少40条训练记录；部署门槛是至少30条留出评价、3个来源组且MAE低于折内均值基线。只对训练化合物Tanimoto相似度≥0.5的未测成分输出预测。误差带是留出绝对误差的90%分位，不是校准置信区间。INSR及AKT1-Ki缺少足够评价，不输出预测。</p>'
    measured=[r for r in chem.data['affinities'] if r['kind']=='measured']
    body+=table(['PubChem CID','靶点/终点','pK','ChEMBL活性ID','来源文献'],[[r['cid'],r['target']+' '+r['endpoint'],round(r['pK'],4),r['activity_ids'],r['document_ids']] for r in measured])
    body+='<h2>3. 实际用于决策的计算</h2><p>索引 clip((pK−4)/5,0,1) 仅为单调的工程证据评分。实测用原始pK；预测用误差带下界。每味药、每个靶点终点取已知成分的最大索引；集合再次取最大，避免重复成分路径累加。Ki和Kd独立成列，完全缺失的请求列不进入分母。缺失与已知零值有不同掩码。此聚合没有药材浓度、吸收率、协同或治疗方向依据。</p>'
    body+='<p>完整目标 = 0.65×原关联目标 + 0.35×分子索引 − 0.35×0.02×药材数。原关联目标已含0.02/味成本，因此总成本仍为0.02/味。三个靶点是明确的糖尿病分子研究上下文，未伪造它们与24症状的显著关联。与仅报告亲和度的旧版本不同，分子覆盖、缺失掩码及边际增益实际进入状态、动作特征、训练奖励和推理。</p>'+picture('molecular_matrix')
    body+='<h2>4. 新策略实验与消融</h2><p>训练/验证/测试分别为320/80/160条请求，对应160/40/80组症状；避开前轮全部症状组合。每组两个工程约束配置，不是临床病例或药理禁忌标签。训练集精确求解生成监督，35轮预训练、15轮价值预训练、100次PPO更新；三种网络各三个随机种子。检查点和最终完整训练模型都只按验证结果选择，随后统一测试。</p>'+picture('planning_comparison')
    body+=table(['方法（三种子均值）','综合目标','距精确最优','约束违规','与贪心集合不同次数'],[
        [k,round(r['all']['objective'],6),round(r['all']['mean_gap_to_optimal'],6),r['all']['hard_violations'],r['all']['different_from_greedy']]
        for k,r in planning['summary'].items()])
    body+='<p>分子证据改变单步贪心候选集合：'+esc(analysis['chemistry_changes_selection']['changed'])+'/160条。所有消融候选均按同一完整目标重算。部署完整训练模型：'+esc(analysis['deployment_selection']['final_ppo_run'])+'，PPO第100次更新。验证选中的最佳检查点另行保留；−1代表初始先验，0代表监督预训练，不能称作PPO训练收益。</p>'
    stats=analysis['final_ppo_vs_greedy_descriptive'];low,high=stats['query_bootstrap_95ci']
    body+=f'<p>完整训练模型相对同目标贪心的平均差为{stats["mean_difference"]:.6f}，按80个症状组进行配对自助抽样的95%区间为[{low:.6f}, {high:.6f}]。本轮原始KAN/PPO低于贪心，未证明整体优势。贪心核对模式独立生成两条轨迹再择优，本轮采用网络轨迹{analysis["guard_used_network"]}/160条，其改善不能归因于纯PPO。</p>'
    body+='<h2>5. 软件使用与尚需完成的验证</h2><p>“辅助分析→选择策略”可选分子证据增益、分子KAN/PPO、分子KAN/PPO＋贪心核对；解释页展示成分、CID、结构、靶点、实测/预测、误差与选择轨迹。保存后来源随患者记录持久化，JSON与HTML导出保留完整链路。“研究复现→成分结构与分子推荐”提供结构查询、目标选择、约束试算和实验表。</p>'
    body+='<p>仍缺少广泛的药材成分实测亲和度、成分在制剂及人体中的暴露浓度、作用方向、独立药效/安全验证。原论文临床队列、体质适配标签、文献频率PMI/TF-IDF边权及完整临床终点评价尚未补齐。当前采用可审计的分子索引工程目标，尚未校准原文结合概率及症状收益系数，也不等同于原文指数饱和奖励的完整复现。本轮没有复现论文86%准确率、0.83 F1或0.67 TCS，也没有据结果选择性改写基线。</p>'
    body+='<h2>来源与复算</h2><p><a href="https://academic.oup.com/nar/article/47/D1/D1110/5150228">SymMap作者论文</a>；<a href="http://www.symmap.org/download/">SymMap官方目录</a>；<a href="https://old.tcmsp-e.com/">TCMSP</a>；<a href="https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial">PubChem PUG REST</a>；<a href="https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions">ChEMBL测定与质量说明</a>。</p><p>完整参数、逐行结果、请求哈希、身份排除及模型权重见同目录JSON和models/planning_v5；运行scripts/verify_v5.py离线核查。数据来源条款与代码许可分别记录于DATA_LICENSE.txt。</p>'
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DMease 分子链路研究报告</title><style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1120px;margin:36px auto;padding:0 24px;color:#18232b;line-height:1.75}h1{font-size:28px}h2{font-size:21px;margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #9ba4aa;padding:7px;text-align:left}th{background:#f1f3f4}img{max-width:100%;height:auto}p{overflow-wrap:anywhere}@media print{tr,img{break-inside:avoid}h2{break-after:avoid}}</style><body>'+body+'</body></html>'
    (OUT/'report.html').write_text(document,encoding='utf8',newline='\n')
    clean_svgs()
    print('Report and3 figure pairs generated',flush=True)


if __name__=='__main__':main()

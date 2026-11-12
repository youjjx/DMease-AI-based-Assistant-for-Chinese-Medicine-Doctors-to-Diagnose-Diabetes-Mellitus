"""Inspectable molecular evidence and connected recommendation workflow."""
import json
from pathlib import Path

import streamlit as st

from .evidence_policy import canonical_hash
from .molecular_planning import MolecularEvidence
from .molecular_runtime import recommend


def display_result(result,key='molecular'):
    a,b,c=st.columns(3)
    a.metric('综合检索目标',f"{result['objective']:.4f}")
    b.metric('分子证据索引',f"{result['molecular_index']:.4f}")
    model=result.get('model',{})
    actual={'network':'KAN网络','greedy':'证据增益'}.get(result['used'],result['used'])
    if result['used']=='network' and model.get('iteration')==-1:actual='初始增益先验'
    c.metric('实际采用',actual)
    if model:st.caption(f"模型：{model['run']}；迭代：{model['iteration']}；模型选择依据验证集。")
    if model.get('iteration')==-1:st.caption('验证集选中了训练前的初始增益先验，此结果不计作KAN/PPO学习收益。')
    elif model.get('iteration')==0:st.caption('本检查点为监督预训练结束、PPO更新之前。')
    if result.get('guard_comparison'):st.write('两条独立轨迹核对',result['guard_comparison'])
    st.write('候选顺序：'+(' → '.join(result['selected']) or '无符合条件的候选'))
    st.dataframe([{'步骤':r['step'],'药材':r['herb'],'新增目标值':r['marginal_evidence_gain'],
                   '选择后分子索引':r['molecular_index_after'],'策略概率':r['policy_probability']} for r in result['steps']],hide_index=True,width='stretch')
    if result['unavailable_channels']:
        st.info('当前证据库缺少可用记录的请求通道：'+'、'.join(result['unavailable_channels'])+'。缺失项不作为弱结合或无效证据。')
    st.caption('索引用于比较可追溯的分子证据。它不代表结合概率、药材体内暴露、激动/抑制方向或临床疗效。预测使用误差带下界排序；误差带未经独立覆盖率校准。')
    if result.get('out_of_training_configuration'):
        st.caption('本次靶点、权重、数量或证据模式与固定训练条件不同；下方实验指标不适用于这组配置。')
    if result['traces']:
        st.dataframe([{'药材':r['herb'],'成分':r['compound_name'],'TCMSP':r['tcmsp_id'],'PubChem CID':r['cid'],
            '靶点':r['target'],'UniProt':r['uniprot'],'终点':r['endpoint'],'证据':'实测' if r['kind']=='measured' else '预测',
            'pK':r['pK'],'排序使用pK':r['ranking_pK'],'索引':r['index'],
            '结构最近相似度':r.get('nearest_tanimoto'),'文献':', '.join(r.get('document_ids',[]))} for r in result['traces']],hide_index=True,width='stretch')
    with st.expander('查看每一步的局部敏感度及来源'):
        st.caption('症状梯度是固定派生动作特征时，模型对输入状态的偏导数，不是医学因果解释。')
        st.json(result)
    st.download_button('导出分子推荐完整链路',json.dumps(result,ensure_ascii=False,indent=2),'dmease_molecular_trace.json','application/json',key=key+'_download')


def show(service,root):
    root=Path(root);folder=root/'experiments/v5';path=folder/'molecular_bundle.json'
    if not path.exists():st.info('分子证据尚未构建，请按项目说明完成第五轮数据准备。');return
    try:chem=MolecularEvidence(path,service.evidence)
    except ValueError as error:st.error(str(error));return
    st.caption('药材—成分—结构—靶点Ki/Kd—分子证据索引—KAN/PPO；各条边均可追溯。')
    view=st.selectbox('分子研究项目',['链路查询','推荐与解释','实验结果'])
    if view=='链路查询':
        a,b,c=st.columns(3);audit=chem.data['audit']
        a.metric('取得结构的成分',audit['unique_compounds']);b.metric('匹配实测记录',audit['measured_compound_target_endpoint_records']);c.metric('域内预测记录',audit['predicted_compound_target_endpoint_records'])
        name=st.selectbox('查询药材',service.graph.herb_names);herb_id=service.evidence.herbs[service.evidence.hnames[name]]['id']
        links=chem.data['herb_compounds'].get(herb_id,[])
        if not links:st.info('该药材当前没有已核验的完整成分—结构链路，未使用相似药名补齐。');return
        st.write(f'本药材有 {len(links)} 条取得结构的编号连接。')
        labels={r['cid']:f"{r['compound_name']} · CID {r['cid']}" for r in links}
        cid=st.selectbox('查询成分',list(labels),format_func=labels.get);structure=chem.data['compounds'][str(cid)]
        st.write('完整InChIKey：',structure['InChIKey']);st.code(structure['parent_smiles'],language=None)
        st.markdown(f"[查看PubChem原始结构记录]({structure['source_url']})")
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            st.image(Draw.MolToImage(Chem.MolFromSmiles(structure['parent_smiles']),size=(540,300)),caption='由已核验SMILES绘制的二维结构')
        except ImportError:st.caption('安装项目affinity依赖后可显示二维结构图。')
        rows=chem.compound_records.get(cid,[])
        if rows:st.dataframe(rows,hide_index=True,width='stretch')
        else:st.info('该成分结构已取得，但当前三个靶点的Ki/Kd实测匹配与可用域内预测均缺失。结构存在不等于亲和度已知。')
        link=next(r for r in links if r['cid']==cid)
        st.graphviz_chart('digraph {rankdir=LR;'+f'h[label={json.dumps(name,ensure_ascii=False)}];c[label={json.dumps(link["tcmsp_id"])}];s[label="PubChem CID {cid}"];h->c;c->s;'+
            ''.join(f't{i}[label={json.dumps(r["target"]+" "+r["endpoint"]+" / "+r["kind"])}];s->t{i};' for i,r in enumerate(rows))+'}')
        with st.expander('查看数据库交叉标识与来源'):st.json({'link':link,'structure':structure})
    elif view=='推荐与解释':
        names=st.multiselect('研究症状',service.graph.symptom_names,default=service.graph.symptom_names[:3])
        targets=st.multiselect('研究靶点',list(chem.data['targets']),default=list(chem.data['targets']))
        forbidden=st.multiselect('排除药材',service.graph.herb_names)
        pair_text=st.text_area('配对排除（每行两味药材，用顿号分隔）')
        budget=st.slider('本次候选上限',1,12,8)
        strategy=st.selectbox('分子策略',['molecular_kan','molecular_greedy','molecular_guarded','molecular_selected'],
            format_func=lambda x:{'molecular_kan':'KAN/PPO完整训练模型','molecular_greedy':'同目标贪心','molecular_guarded':'KAN/PPO＋独立贪心核对','molecular_selected':'验证集选中的检查点'}[x])
        mode=st.selectbox('亲和度依据',['combined','measured_only'],format_func=lambda x:'实测＋域内预测' if x=='combined' else '仅实测')
        weight=st.slider('分子证据权重',0.,1.,.35,.05)
        pairs=[[v.strip() for v in line.replace('，','、').replace(',','、').split('、')] for line in pair_text.splitlines() if line.strip()]
        task={'symptoms':dict.fromkeys(names,1.),'molecular_targets':targets,'forbidden':forbidden,'forbidden_pairs':pairs,'budget':budget}
        signature=canonical_hash([task,strategy,mode,weight,chem.fingerprint])
        if st.button('执行分子链路推荐',disabled=not names or not targets):
            try:
                if any(len(p)!=2 or len(set(p))!=2 or not set(p)<=set(service.graph.herb_names) for p in pairs):raise ValueError('每行请填写两味不同的当前药材。')
                result=recommend(service.evidence,task,root,strategy,weight,mode)
                st.session_state['v5_molecular_result']=(signature,result)
            except (ValueError,OSError) as error:st.error(str(error))
        stored=st.session_state.get('v5_molecular_result')
        if stored and stored[0]==signature:display_result(stored[1])
    else:
        data=json.loads((folder/'affinity_evaluation.json').read_text(encoding='utf8'))
        st.write('结构预测：文献与分子骨架联合分组留出')
        st.dataframe([{'通道':k,'训练资料条数':r['rows'],'实际留出评价条数':r['evaluated_rows'],
                       '来源组':r['evaluated_components'],'MAE':r.get('model',{}).get('MAE_pK'),
                       '均值基线MAE':r.get('baseline',{}).get('MAE_pK'),'允许域内预测':r['deployable']} for k,r in data['strata'].items()],hide_index=True,width='stretch')
        st.caption('沿用已探索过的数据与超参数，属于开发性稳健性检查。样本不足的通道不产生预测。')
        if (folder/'planning_summary.json').exists():
            planning=json.loads((folder/'planning_summary.json').read_text(encoding='utf8'))
            st.write('相同分子目标下的策略对照')
            st.dataframe([{'方法':name,**row['all']} for name,row in planning['summary'].items()],hide_index=True,width='stretch')
            st.caption('160条测试检索请求，来自80组症状；工程约束不代表临床禁忌。消融候选统一按完整目标重算。')
        if (folder/'report.html').exists():st.download_button('下载第五轮研究报告',(folder/'report.html').read_bytes(),'DMease_分子链路研究报告.html','text/html')

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from dmease import DMeaseService
from dmease.auth import AuthStore
from dmease.exports import SOFTWARE_NAME, VERSION, csv_export, json_export, report_html
from dmease.types import PatientProfile
from dmease.public_data import PublicEvidenceStore

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DMEASE_DATA_DIR", str(ROOT / "runtime")))
DEFAULT_CHECKPOINT = ROOT / "models/public_evidence_kan_ppo.pt"
CHECKPOINT = Path(os.environ.get("DMEASE_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
BUNDLE = Path(os.environ.get("DMEASE_EVIDENCE_BUNDLE", str(ROOT / "data/processed/evidence_bundle.json")))
PAGES = ["首页", "辅助分析", "患者记录", "知识图谱", "研究复现", "系统与账号"]
STEPS = ["1 资料录入", "2 症状与证候", "3 候选与解释", "4 约束与保存"]
CONSTITUTIONS = ["平和质", "气虚质", "阴虚质", "阳虚质", "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"]

st.set_page_config(page_title=SOFTWARE_NAME, page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:2rem;padding-bottom:2rem;max-width:1250px}
h1{font-size:1.9rem!important;letter-spacing:.01em}h2{font-size:1.35rem!important}
[data-testid="stSidebar"]{background:#f4f7f8}
[data-testid="stMetric"]{background:#f5f8f8;padding:12px;border-radius:6px}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_auth(path):
    return AuthStore(Path(path) / "accounts.db")


@st.cache_resource
def load_service(path, checkpoint, modified, bundle, bundle_modified):
    return DMeaseService(bundle, Path(path) / "patients.db", checkpoint)


def go(page, step=None):
    st.session_state["pending_navigation"] = page
    if step:
        st.session_state["pending_step"] = step


def clear_session(message=""):
    for key in list(st.session_state):
        del st.session_state[key]
    if message:
        st.session_state["login_message"] = message


def authenticate(auth):
    account = st.session_state.get("account")
    if account:
        expired = time.time() - st.session_state.get("last_activity", 0) > 1800
        changed = auth.revision(account) != st.session_state.get("credential_revision")
        if not expired and not changed:
            st.session_state["last_activity"] = time.time()
            return account
        clear_session("登录已过期或账号密码已更新，请重新登录。")
    st.title(SOFTWARE_NAME)
    st.caption(f"{VERSION}  |  知识图谱 · 症状解析 · 证候判断 · 中药候选解释")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("欢迎使用 DMease")
        st.write("登录后进入工作台，完成资料录入、辅助分析、记录管理与知识查询。")
        st.markdown("1. 录入患者资料与中文主诉\n2. 查看证候依据与中药候选\n3. 核对约束、保存和导出结果")
        st.info("系统使用可追溯的公开数据库关联进行辅助检索。结果须由专业人员核查。")
    with right:
        if not auth.initialized():
            st.subheader("首次使用 · 创建账号")
            st.caption("首次部署请在受控环境下完成初始化；系统不提供默认密码。")
            with st.form("setup", clear_on_submit=True):
                username = st.text_input("账号", max_chars=32)
                password = st.text_input("密码", type="password", max_chars=128)
                repeat = st.text_input("确认密码", type="password", max_chars=128)
                st.caption("账号3—32位；密码12—128位，包含英文字母和数字。")
                submitted = st.form_submit_button("创建账号", type="primary")
            if submitted:
                try:
                    if password != repeat:
                        raise ValueError("两次输入的密码不一致。")
                    auth.bootstrap(username, password)
                    st.session_state["login_message"] = "账号已创建，请登录。"
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        else:
            st.subheader("账号登录")
            if st.session_state.get("login_message"):
                st.info(st.session_state.pop("login_message"))
            with st.form("login", clear_on_submit=True):
                username = st.text_input("账号", max_chars=32)
                password = st.text_input("密码", type="password", max_chars=128)
                submitted = st.form_submit_button("登录", type="primary", width="stretch")
            st.caption("连续5次验证失败后暂停验证60秒；30分钟无操作后需重新登录。")
            if submitted:
                revision = auth.authenticate(username, password)
                if revision is None:
                    st.error("账号或密码不正确，或账号暂时锁定。请核对后重试。")
                else:
                    clear_session()
                    st.session_state.update(account=username.strip(), credential_revision=revision,
                                            last_activity=time.time(), navigation="首页")
                    st.rerun()
    st.stop()


def exports(payload, prefix="current"):
    c1, c2, c3 = st.columns(3)
    c1.download_button("导出完整 JSON", json_export(payload), "dmease_analysis.json", "application/json", key=prefix+"json")
    c2.download_button("导出候选 CSV", csv_export(payload["recommendations"]), "dmease_candidates.csv", "text/csv", key=prefix+"csv")
    c3.download_button("下载打印报告", report_html(payload), "dmease_report.html", "text/html", key=prefix+"html")
    st.caption("打印报告下载后用浏览器打开，可打印或另存为PDF。导出内容对应当前显示的分析结果。")


def start_new():
    st.session_state["draft"] = {}
    st.session_state["draft_symptoms"] = []
    st.session_state["draft_top_k"] = 8
    for key in ["analysis_result", "saved_id", "parent_record_id"]:
        st.session_state.pop(key, None)
    st.session_state["form_revision"] = st.session_state.get("form_revision", 0) + 1
    go("辅助分析", STEPS[0])


def home(service, owner):
    st.subheader("工作台")
    st.write("从新建分析开始，或从患者记录继续既往工作。")
    summary = service.graph.summary()
    a,b,c,d = st.columns(4)
    a.metric("已保存分析", service.database.count(owner))
    b.metric("标准症状", summary["symptoms"])
    c.metric("证候类型", summary["syndromes"])
    d.metric("中药实体", summary["herbs"])
    st.caption(f"当前排序模式：{service.recommender.mode} · 知识版本：{summary['version']}")
    a,b,c = st.columns(3)
    a.button("新建辅助分析", type="primary", on_click=start_new, width="stretch")
    b.button("查询患者记录", on_click=go, args=("患者记录",), width="stretch")
    c.button("浏览知识图谱", on_click=go, args=("知识图谱",), width="stretch")
    st.subheader("最近保存")
    rows = service.database.list(5, owner=owner)
    if rows:
        st.dataframe(record_frame(rows), hide_index=True, width="stretch")
    else:
        st.info("尚无保存记录。完成分析后，在“约束与保存”中保存结果。")
    st.subheader("操作流程")
    st.write("资料录入 → 症状与证候 → 候选与解释 → 约束核对 → 保存或导出 → 历史记录复用")
    st.caption("患者标识建议采用匿名编号。候选分数和证候支持度用于解释计算过程，不代表临床疗效。")


def input_form(service):
    draft = st.session_state.get("draft", {})
    rev = st.session_state.get("form_revision", 0)
    with st.form(f"patient_form_{rev}"):
        left, right = st.columns(2)
        with left:
            name = st.text_input("患者标识 *", draft.get("name", ""), max_chars=80, placeholder="例如：样例患者001")
            age = st.number_input("年龄", 1, 120, int(draft.get("age", 50)))
            sexes = ["未说明", "男", "女"]
            sex = st.selectbox("性别", sexes, index=sexes.index(draft.get("sex", "未说明")))
            constitution = st.selectbox("体质", CONSTITUTIONS, index=CONSTITUTIONS.index(draft.get("constitution", "平和质")))
            pregnant = st.checkbox("妊娠期", draft.get("pregnant", False))
        with right:
            allergies = st.text_input("已知过敏", "、".join(draft.get("allergies", [])), max_chars=300, placeholder="中药名称，用逗号或顿号分隔")
            conditions = st.text_input("合并情况", "、".join(draft.get("conditions", [])), max_chars=300)
            medications = st.text_input("当前用药", "、".join(draft.get("medications", [])), max_chars=300)
            notes = st.text_area("补充说明", draft.get("notes", ""), height=105, max_chars=2000)
        complaint = st.text_area("主诉与症状描述 *", draft.get("complaint", ""), height=100, max_chars=4000)
        selected = st.multiselect("补充勾选症状", service.graph.symptom_names, default=[x for x in st.session_state.get("draft_symptoms", []) if x in service.graph.symptom_names])
        forbidden = st.multiselect("本次明确禁用中药", service.graph.herb_names, default=[x for x in draft.get("forbidden_herbs", []) if x in service.graph.herb_names])
        pair_text = st.text_area("本次配伍排除", "\n".join("、".join(x) for x in draft.get("forbidden_pairs", [])),
                                placeholder="每行填写两味当前范围内的中药，用顿号或逗号分隔；填写后不会同时入选。", height=80)
        strategy = st.selectbox("选择策略", ["默认（依据验证集表现）", "公开证据增益", "KAN/PPO对照"])
        top_k = st.slider("候选中药数量上限", 1, 12, int(st.session_state.get("draft_top_k", 8)))
        submitted = st.form_submit_button("开始辅助分析", type="primary", width="stretch")
        saved_draft = st.form_submit_button("保存输入草稿")
    st.caption("切换页面前可保存输入草稿；草稿保留在当前会话，刷新或退出后清除。修改后须重新分析，历史记录复用后保存为新记录。")
    if submitted or saved_draft:
        split = lambda value: [x.strip() for x in value.replace("，", ",").replace("、", ",").split(",") if x.strip()]
        draft = dict(name=name.strip(), age=age, sex=sex, constitution=constitution,
                     pregnant=pregnant, allergies=split(allergies), conditions=split(conditions),
                     medications=split(medications), forbidden_herbs=forbidden,
                     forbidden_pairs=[split(line) for line in pair_text.splitlines() if line.strip()],
                     notes=notes.strip(), complaint=complaint.strip())
        st.session_state.update(draft=draft, draft_symptoms=selected, draft_top_k=top_k)
        # A submitted edit invalidates every previous result, including when validation fails.
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("saved_id", None)
        if saved_draft:
            st.success("输入草稿已保留在当前会话，重新分析后可保存结果。")
            return
        if not name.strip():
            st.error("请填写患者标识。")
        elif not complaint.strip() and not selected:
            st.error("请填写主诉，或至少勾选一个标准症状。")
        elif pregnant and sex == "男":
            st.error("性别与妊娠状态不一致，请核对。")
        else:
            try:
                result = service.analyze(PatientProfile(**draft), selected, top_k,
                                         strategy={"默认（依据验证集表现）": None, "公开证据增益": "greedy", "KAN/PPO对照": "kan"}[strategy])
            except ValueError as error:
                st.error(str(error))
                return
            if not any(x.present for x in result.parsed_symptoms):
                st.warning("未识别到阳性症状。请补充主诉或勾选标准症状后重新分析。")
            else:
                st.session_state["analysis_result"] = result
                st.session_state["pending_step"] = STEPS[1]
                st.rerun()


def analysis(service, owner):
    st.subheader("辅助分析")
    if "pending_step" in st.session_state:
        st.session_state["analysis_step"] = st.session_state.pop("pending_step")
    step = st.radio("分析步骤", STEPS, key="analysis_step", horizontal=True)
    if st.session_state.get("parent_record_id"):
        st.caption(f"正在复用记录 #{st.session_state['parent_record_id']}，本次保存将建立新版本。")
    if step == STEPS[0]:
        input_form(service)
        return
    result = st.session_state.get("analysis_result")
    if not result:
        st.info("请先在“资料录入”中完成一次有效分析。")
        st.button("返回资料录入", on_click=go, args=("辅助分析", STEPS[0]))
        return
    st.caption(f"患者：{result.patient.name} · 分析时间：{result.created_at} · {result.metadata['ranking_mode']}")
    if step == STEPS[1]:
        columns = st.columns(max(1, len(result.syndromes)))
        for col, syndrome in zip(columns, result.syndromes):
            col.metric(syndrome.name, f"{syndrome.confidence:.1%}")
        if not result.syndromes:
            st.info("当前阳性症状没有已取得的证候关联，仍可查看药材检索结果与证据缺口。")
        st.caption("展示前3个证候；支持度按全部候选归一化，三项之和可能小于100%，不代表临床诊断概率。")
        st.dataframe([{"标准症状": x.canonical, "状态": "阳性" if x.present else "否定", "原文证据": x.evidence}
                      for x in result.parsed_symptoms], hide_index=True, width="stretch")
        for syndrome in result.syndromes:
            with st.expander(syndrome.name + " · 判断依据", expanded=syndrome is result.syndromes[0]):
                st.write(syndrome.rationale)
                st.caption("尚缺关键线索：" + ("、".join(syndrome.missing_key_symptoms) or "无"))
        st.button("下一步：候选与解释", on_click=go, args=("辅助分析", STEPS[2]), type="primary")
    elif step == STEPS[2]:
        if not result.recommendations:
            st.info("在当前约束下没有可展示的候选中药。请前往“约束与保存”查看排除原因。")
        else:
            st.dataframe([{"步骤": x.rank, "中药": x.name, "新增证据增益": round(x.score, 4)}
                          for x in result.recommendations], hide_index=True, width="stretch")
            chosen = st.selectbox("选择候选查看解释", [x.name for x in result.recommendations])
            herb = next(x for x in result.recommendations if x.name == chosen)
            st.caption("排序依据公开数据库的关联覆盖；不能将入选顺序解释为处方或疗效优劣。")
            for path in herb.evidence_paths:
                st.write("• " + path)
            for warning in herb.warnings:
                st.caption(warning)
            evidence_store = PublicEvidenceStore(ROOT / "data/processed/public_evidence.sqlite")
            if evidence_store.available:
                with st.expander("核对公开数据库中的成分与靶点"):
                    evidence = evidence_store.herb_paths(herb.name)
                    if evidence:
                        st.dataframe([{k: v for k, v in row.items() if k != "provenance"} for row in evidence],
                                     hide_index=True, width="stretch")
                        st.caption("来源：TCMSP 2.3。按药材名称精确匹配，最多显示20条；数据库关联用于核查，不代表疗效或结合概率。")
                        st.download_button("导出此药材公开依据", json_export(evidence), "herb_evidence.json", "application/json")
                    else:
                        st.info("当前公开数据索引未收录该名称对应的完整路径。")
        trace = result.metadata.get("trace", {})
        st.write("逐步选择过程")
        st.dataframe([{"步骤": x["step"], "中药": x["herb"], "新增证据增益": x["marginal_evidence_gain"],
                       "策略概率": x.get("policy_probability"), "未覆盖证据量": sum(x["residual_evidence"].values())}
                      for x in trace.get("steps", [])], hide_index=True, width="stretch")
        st.caption("停止原因：" + trace.get("stop_reason", "未记录"))
        if trace.get("strategy_notice"):
            st.warning(trace["strategy_notice"])
        if result.metadata.get("evidence_gaps"):
            st.warning("缺少可用药材或靶点关系的症状：" + "、".join(result.metadata["evidence_gaps"]))
        with st.expander("查看策略梯度与完整轨迹"):
            st.caption("梯度是当前策略对未覆盖症状状态的局部敏感度，不是医学因果解释；证据增益策略不产生梯度。")
            st.json(trace)
        st.button("下一步：约束与保存", on_click=go, args=("辅助分析", STEPS[3]), type="primary")
    else:
        st.subheader("安全约束核对")
        if result.excluded_herbs:
            st.dataframe([{"排除中药": herb, "触发原因": "；".join(reasons)} for herb, reasons in result.excluded_herbs.items()],
                         hide_index=True, width="stretch")
        else:
            st.info("当前输入未命中已配置的排除规则；这不等于完成临床安全评估。")
        st.caption("输入过敏项：" + ("、".join(result.patient.allergies) or "未填写") + "；当前用药：" + ("、".join(result.patient.medications) or "未填写"))
        for note in result.safety_notes:
            st.write(note)
        confirmed = st.checkbox("已核对本次输入、约束提示与结果使用范围", key="reviewed_"+result.created_at)
        if st.button("保存本次分析", type="primary", disabled=not confirmed or bool(st.session_state.get("saved_id"))):
            st.session_state["saved_id"] = service.database.save(result, owner, st.session_state.get("parent_record_id"))
            st.rerun()
        if st.session_state.get("saved_id"):
            st.success(f"已保存为记录 #{st.session_state['saved_id']}；重复点击不会生成重复记录。")
        exports(result.to_dict())
        st.button("前往患者记录", on_click=go, args=("患者记录",))


def record_frame(rows):
    return pd.DataFrame(rows).rename(columns={"id":"记录号", "patient_name":"患者标识", "created_at":"分析时间", "complaint":"主诉",
                                              "primary_syndrome":"首选证候", "parent_record_id":"来源记录"})


def reuse_record(payload, chosen):
    st.session_state.update(draft=payload["patient"], parent_record_id=chosen,
                            draft_symptoms=payload.get("metadata", {}).get("selected_symptoms", []),
                            draft_top_k=payload.get("metadata", {}).get("top_herbs", 8))
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("saved_id", None)
    st.session_state["form_revision"] = st.session_state.get("form_revision", 0) + 1
    go("辅助分析", STEPS[0])


def records(service, owner):
    st.subheader("患者记录")
    a,b = st.columns([2,1])
    query = a.text_input("搜索患者标识或主诉", max_chars=100)
    syndrome = b.selectbox("筛选首选证候", ["全部"] + service.graph.syndrome_names)
    rows = service.database.list(500, owner=owner, query=query, syndrome="" if syndrome == "全部" else syndrome)
    st.caption(f"当前账号共 {service.database.count(owner)} 条记录；本次显示 {len(rows)} 条，最多显示最近500条。")
    if not rows:
        st.info("没有匹配记录。可调整筛选条件，或前往辅助分析创建记录。")
        return
    st.dataframe(record_frame(rows), hide_index=True, width="stretch")
    st.download_button("导出当前列表 CSV", csv_export(record_frame(rows).to_dict("records")), "dmease_records.csv", "text/csv")
    labels = {r["id"]: f"记录 #{r['id']} · {r['patient_name']} · {r['created_at']}" for r in rows}
    chosen = st.selectbox("选择记录", list(labels), format_func=labels.get)
    payload = service.database.get(chosen, owner=owner)
    if not payload:
        st.warning("记录已变化，请刷新列表。")
        return
    p = payload["patient"]
    st.write(f"**患者资料**　{p['name']} / {p['age']}岁 / {p['sex']} / {p['constitution']}")
    st.write("主诉：" + p["complaint"])
    st.write("首选证候：" + (payload["syndromes"][0]["name"] if payload["syndromes"] else "无"))
    parent_id = next(r["parent_record_id"] for r in rows if r["id"] == chosen)
    if parent_id:
        original = service.database.get(parent_id, owner=owner)
        with st.expander(f"与来源记录 #{parent_id} 比较", expanded=True):
            if original:
                comparison = []
                for key, label in (("complaint", "主诉"), ("allergies", "过敏项"), ("medications", "当前用药"), ("notes", "补充说明")):
                    before, after = original["patient"][key], p[key]
                    if before != after:
                        comparison.append({"字段": label, "来源记录": str(before), "当前记录": str(after)})
                if comparison:
                    st.dataframe(comparison, hide_index=True, width="stretch")
                else:
                    st.caption("主诉、过敏项、当前用药和补充说明未改变。")
                st.write("原首选证候：" + original["syndromes"][0]["name"] + "；本次首选证候：" + payload["syndromes"][0]["name"])
            else:
                st.caption("来源记录已删除；当前记录保留完整分析快照。")
    with st.expander("查看完整结果与证据"):
        st.json(payload)
    exports(payload, "record")
    st.button("复用资料并重新分析", type="primary", on_click=reuse_record, args=(payload, chosen))
    with st.expander("删除此记录"):
        agreed = st.checkbox(f"确认永久删除记录 #{chosen}", key=f"delete_confirm_{chosen}")
        if st.button("确认删除", disabled=not agreed, key=f"delete_{chosen}"):
            service.database.delete(chosen, owner=owner)
            if st.session_state.get("parent_record_id") == chosen:
                st.session_state.pop("parent_record_id", None)
            if st.session_state.get("saved_id") == chosen:
                st.session_state.pop("saved_id", None)
            st.rerun()


def graph_page(service):
    graph = service.graph
    st.subheader("知识图谱")
    view = st.radio("图谱视图", ["证候关联", "机制路径", "知识子图", "公开数据检索"], horizontal=True)
    if view == "公开数据检索":
        public_evidence_page()
        return
    st.caption(f"知识版本：{graph.summary()['version']} · {graph.summary()['relations']}条取得的关系 · 公开数据库证据")
    if view in ("证候关联", "知识子图"):
        name = st.selectbox("选择证候", graph.syndrome_names)
        syndrome = graph.syndrome(name)
        st.write(syndrome["description"])
        herbs = graph.herbs_for_syndrome(name)
        if view == "证候关联":
            a,b = st.columns(2)
            a.dataframe([{"症状": n, "公开关联": "已收录"} for n in syndrome["symptoms"]], hide_index=True, width="stretch")
            b.dataframe([{"共享症状关联的中药": h["herb"]["name"], "共享症状数": h["weight"]} for h in herbs], hide_index=True, width="stretch")
            st.caption("右侧为经共同症状连接的药材，不表示已验证的证候治疗关系。")
        else:
            quote = lambda x: json.dumps(x, ensure_ascii=False)
            edges = [f'{quote(s)} -> {quote(name)} [label="支持"];' for s in syndrome["symptoms"]]
            edges += [f'{quote(name)} -> {quote(h["herb"]["name"])} [label="共享症状路径"];' for h in herbs]
            st.graphviz_chart('digraph { rankdir=LR; node [shape=box,fontname="Microsoft YaHei"]; edge [fontname="Microsoft YaHei"]; ' + ''.join(edges) + '}', width="stretch", height=430)
    else:
        herb = st.selectbox("选择中药", graph.herb_names)
        info = graph.herb(herb)
        st.caption("展示已取得且满足筛选条件的公开靶点关联。")
        paths = graph.mechanism_paths(herb)
        if paths:
            for path in paths:
                st.write(path)
        else:
            st.info("当前范围未取得满足筛选条件的共同靶点关系。")
        with st.expander("查看知识库中的约束信息", expanded=True):
            st.write("公开关联库不包含完整禁忌与相互作用规则。当前自动执行填写的过敏、明确禁用及妊娠停止规则；其他适用性需专业核查。")


def public_evidence_page():
    st.caption("查询官方公开数据的实体与蛋白关联，查看版本、来源和导入范围。")
    registry_path = ROOT / "data/public_registry.json"
    store = PublicEvidenceStore(ROOT / "data/processed/public_evidence.sqlite")
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf8"))
        st.caption("以下为最近一次公开数据导入报告；本机索引可用性在下方单独检查。")
        labels = {"human_network_imported": "人类蛋白网络已导入", "official_entity_catalogues_imported": "实体及标识表已导入",
                  "official_browse_catalogues_imported": "公开浏览目录已导入", "failed": "导入失败"}
        scope_labels = {"string": "12.0版人类蛋白网络，保留综合分数≥0.7的去重关联",
                        "symmap": "1.0/2.0版公开实体与标识表；关联关系尚待补充",
                        "tcmsp": "四类公开目录及与系统药材名称精确匹配的成分—靶点详情"}
        st.dataframe([{"数据库": name.upper(), "状态": labels.get(item["status"], item["status"]),
                       "范围": item.get("error", scope_labels.get(name, "见来源报告"))}
                      for name, item in registry["sources"].items()], hide_index=True, width="stretch")
        with st.expander("查看下载来源与文件校验信息"):
            st.json(registry)
    if not store.available:
        st.info("此部署尚未建立公开数据索引，请按项目说明完成数据初始化。")
        return
    st.info("实际候选流程使用已构建的SymMap/STRING证据版本；本页另可查询完整已下载目录及TCMSP补充依据。")
    source = st.selectbox("公开数据来源", ["全部", "SymMap", "STRING", "TCMSP"])
    query = st.text_input("检索名称或原始标识", placeholder="例如：黄芪、AKT1、INSR")
    rows = store.search(query, "" if source == "全部" else source)
    if not rows:
        st.caption("输入检索词查看结果。未匹配时可尝试英文名或原始数据库标识。")
        return
    st.dataframe([{"来源": row["source"], "版本": row["version"], "类别": row["kind"],
                   "原始标识": row["id"], "名称": row["name"]} for row in rows], hide_index=True, width="stretch")
    chosen = st.selectbox("选择公开条目", range(len(rows)),
                          format_func=lambda i: f"{rows[i]['source']} {rows[i]['version']} · {rows[i]['id']} · {rows[i]['name'][:70]}")
    item = rows[chosen]
    st.json(json.loads(item["payload"]))
    if item["source"] == "STRING":
        edges = store.neighbors(item["id"])
        st.write("高置信度功能关联（综合分数≥0.7；最多显示30条）")
        st.dataframe(edges, hide_index=True, width="stretch")
        st.caption("功能关联可包含间接关联与预测证据，不能直接解释为已验证的物理结合。")
    elif item["source"] == "TCMSP":
        edges = store.neighbors(item["id"])
        st.write("已导入的成分与靶点关联（最多显示30条）")
        st.dataframe(edges, hide_index=True, width="stretch")
        st.caption("仅展示已下载药材详情中的关联；空结果表示当前索引未收录。原始验证标记和预测分数保存在来源字段中。")
    st.download_button("导出当前公开检索结果", json_export(rows), "public_evidence.json", "application/json")


def experimental_evaluation(service):
    folder = ROOT / "experiments/results"
    path = folder / "summary.json"
    if not path.exists():
        st.info("此版本尚未附带扩展实验报告。")
        return
    report = json.loads(path.read_text(encoding="utf8"))
    st.caption("固定公开图谱上的工程实验：464个训练查询、49个验证查询、62个测试查询；三个随机种子。查询不是病例。")
    if report["provenance"]["evidence_fingerprint"] != service.evidence.fingerprint:
        st.warning("实验报告与本机当前证据版本不同，下列结果仅描述报告所记录的版本。")
    names = {"full": "KAN + PPO", "mlp": "MLP + PPO", "without_ppi": "移除PPI",
             "without_direct": "移除直接关系", "without_targets": "移除靶点及PPI",
             "warm_start_only": "仅模仿预训练", "ppo_only": "仅PPO训练"}
    section = st.selectbox("实验项目", ["模型与消融", "约束压力测试", "参数敏感性", "证据缺失", "精确最优解对照", "文本解析回归"])
    if section == "模型与消融":
        st.dataframe([{"方法": names[k], "综合目标均值": v["seed_mean"]["objective"],
                       "种子间标准差": v["seed_std"]["objective"], "平均候选数": v["seed_mean"]["herb_count"],
                       "参数量": v["parameters"]} for k, v in report["ablations"].items()], hide_index=True, width="stretch")
        st.write(f"证据增益基线综合目标：{report['baseline_greedy']['mean']['objective']:.6f}。")
        st.caption("所有候选集合均在同一完整参考图谱和目标下重新评分；当前结果不支持KAN/PPO优于简单基线。模型和默认参数不按测试结果更换。")
        baseline = report["baseline_greedy"]
        st.write(f"靶点证据可用查询：{baseline['target_available_queries']}/62；PPI证据可用查询：{baseline['ppi_available_queries']}/62。")
        st.caption("缺少证据的查询与有证据时的覆盖率分开统计，避免将数据缺失解释为算法失败或疗效为零。")
        with st.expander("配对比较与统计方法"):
            st.dataframe([{"对照": names[k], "完整模型减对照": v["mean_difference"],
                           "区间下限": v["query_bootstrap_95ci"][0], "区间上限": v["query_bootstrap_95ci"][1],
                           "Holm调整p": v["holm_p"]} for k, v in report["full_minus_variant"].items()], hide_index=True)
            st.caption("先平均三个固定种子的每查询差值，再按查询重采样。症状组合存在重叠，独立性未成立；区间和p值只作条件性描述，不是临床统计证据。")
    elif section == "约束压力测试":
        data = json.loads((folder / "constraints.json").read_text(encoding="utf8"))
        labels = {"exclude_first_baseline_choice": "排除基线首选", "exclude_all": "全部排除",
                  "only_one_allowed": "仅允许一种", "forbid_first_two_as_pair": "排除首两味同时入选"}
        st.metric("违反指定约束次数 / 总查询执行次数", f"{report['constraint_violations']} / {report['constraints_evaluated']}")
        st.dataframe([{"场景": labels[r["scenario"]], "策略": r["mode"], "种子": r["seed"],
                       "有效约束查询": r["active_constraint_queries"], "违规次数": r["violation_total"],
                       "平均候选数": r["mean"]["herb_count"]} for r in data["results"]], hide_index=True, width="stretch")
        st.caption("测试的是明确输入的排除规则，规则用于检查程序行为，不构成药物禁忌知识或临床安全性验证。")
    elif section in ("参数敏感性", "证据缺失"):
        filename = "sensitivity.json" if section == "参数敏感性" else "missing_evidence.json"
        data = json.loads((folder / filename).read_text(encoding="utf8"))
        rows = [{**{k: r[k] for k in ("parameter", "value") if k in r},
                 **{k: r[k] for k in ("drop_fraction", "mode", "seed") if k in r},
                 "参考综合目标": r["mean"]["objective"], "平均候选数": r["mean"]["herb_count"]} for r in data["results"]]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("参数仅改变选择过程，评分仍使用完整参考图谱和默认目标。证据缺失实验只删除已观察到的关系；模型使用完整图谱训练的固定权重，不重新训练。")
    elif section == "精确最优解对照":
        data = json.loads((folder / "optimality.json").read_text(encoding="utf8"))
        st.write(f"已检查{report['exact_subsets_evaluated']:,}个可行子集，数量上限为8味。")
        stats = data["exact_minus_greedy"]
        st.metric("最优解相对证据增益的平均目标差", f"{stats['mean_difference']:.6f}")
        st.write(f"62个查询中，最优解更高 / 相同 / 更低：{stats['wins_ties_losses']}（数值容差为0.000001）。")
        st.dataframe([{"查询标识": r["query_id"], "最优参考目标": r["objective"],
                       "候选集合": "、".join(r["selected_herbs"]), "检查子集数": r["enumerated_feasible_subsets"]}
                      for r in data["exact"]], hide_index=True, width="stretch")
        st.caption("这是给定图谱与工程目标的集合最优解，不表示最佳处方。枚举仅适用于当前小规模范围。")
    else:
        before = json.loads((folder / "parser_before.json").read_text(encoding="utf8"))
        after = json.loads((folder / "parser_after.json").read_text(encoding="utf8"))
        st.dataframe([{"版本": label, "用例数": r["cases"], "完全一致数": r["exact_matches"],
                       "实体及状态micro-F1": r["entity_state_micro_f1"]}
                      for label, r in [("修复前", before["summary"]["all"]), ("修复后", after["summary"]["all"])]], hide_index=True)
        st.warning("164条用例由开发过程编写或模板展开，参与了规则修复。这是回归检查，不是独立测试集、专家标注或临床识别准确率。")
        fixed = [r for r in before["rows"] if not r["exact"]]
        with st.expander("查看修复前的错误用例"):
            st.dataframe([{"文本": r["text"], "预期": json.dumps(r["expected"], ensure_ascii=False),
                           "原输出": json.dumps(r["predicted"], ensure_ascii=False)} for r in fixed], hide_index=True)
    st.download_button("下载实验汇总 JSON", json_export(report), "dmease_experiments.json", "application/json")
    html = ROOT / "experiments/report.html"
    if html.exists():
        st.download_button("下载完整实验报告", html.read_bytes(), "DMease_实验报告.html", "text/html")


def research_v2_evaluation(service):
    folder = ROOT / "experiments/v2"
    if not (folder / "summary.json").exists():
        st.info("此版本未附带第二轮实验报告。")
        return
    summary = json.loads((folder / "summary.json").read_text(encoding="utf8"))
    if summary["provenance"]["evidence"] != service.evidence.fingerprint:
        st.warning("第二轮报告与本机证据版本不同，指标只描述报告中的固定版本。")
    section = st.selectbox("改进研究项目", ["策略改进", "真实病例辨证", "通路与靶点一致性", "单次集合优化"])
    names = {"greedy": "证据增益", "legacy": "旧版KAN/PPO", "stable_rbf": "稳定RBF方案", "stable_spline": "三次样条KAN",
             "oracle_spline": "最优集合示范与样条", "oracle_mlp": "最优集合示范与MLP"}
    if section == "策略改进":
        st.caption("464个训练查询；与第一轮不重叠的96个验证和192个测试查询。全部检查点在新测试评价前冻结。")
        st.dataframe([{"方案": names[k], "原始目标": v["raw"]["objective"], "种子标准差": v["raw"]["objective_seed_std"],
                       "相同集合优化后": v["shared_local_search"]["objective"], "原始候选数": v["raw"]["herb_count"]}
                      for k,v in summary["results"].items()], hide_index=True, width="stretch")
        st.info("稳定方案缩小了与贪心的差距，但未超过贪心；稳定RBF三个种子均选择PPO前检查点，收益不能归因于PPO。统一集合优化后，贪心在190/192个查询达到给定目标最优。")
        st.caption("集合搜索使用相同图谱和成本，分别提供给各方法；其收益不算作KAN独有优势。证据目标不表示疗效。")
    elif section == "真实病例辨证":
        path = folder / "supervised_syndrome.json"
        if not path.exists():
            st.info("监督辨证报告尚未生成。")
            return
        data = json.loads(path.read_text(encoding="utf8"))
        st.write(f"TCM-SD：训练{data['train']:,}条、验证{data['dev']:,}条、测试{data['test']:,}条，148类证候。")
        st.dataframe([{"方法": method, "评价范围": group, "记录数": v["records"], "准确率": v["accuracy"],
                       "宏平均F1": v["macro_f1_148_train_labels"], "Kappa": v["kappa"], "前三命中率": v["top3_hit"]}
                      for method, groups in data["evaluation"].items() for group,v in groups.items()], hide_index=True, width="stretch")
        st.caption(f"测试中{data['test_seen_texts_train_or_dev']}条文本与训练或验证重复，已另外报告去重后的结果。源user_id有跨分区重复，其语义未核实，不能宣称患者级独立测试。")
        st.warning("监督文本分类器是独立研究对照，并非KAN/PPO或论文的LLM；数据覆盖一般中医疾病，不能作为原论文50例糖尿病研究的复现。当前临床工作流未自动替换模型。")
        external = json.loads((folder / "external_syndrome.json").read_text(encoding="utf8"))
        st.write("TCMEval-SDT外部审计：200条记录，原图谱前三候选命中2条；335个金标准证候标签中仅28个能按固定文字规则映射。")
        if data["tcmeval_transfer"]:
            transfer = data["tcmeval_transfer"]
            st.write(f"冻结文本分类器跨数据集迁移：首候选命中{transfer['top1_any_gold_hit']:.1%}，前三命中{transfer['top3_any_gold_hit']:.1%}。")
        st.caption("外部集的低表现揭示标签体系和适用范围缺口；本页不展示任何病例原文。")
    elif section == "通路与靶点一致性":
        st.dataframe([{"方案": names[k], "共享49靶点Jaccard": v["raw"]["target_jaccard_shared49"],
                       "全部显著关联靶点Jaccard": v["raw"]["target_jaccard_all_eligible"]} for k,v in summary["results"].items()], hide_index=True, width="stretch")
        st.caption("Jaccard按交集/并集计算。当前使用关联靶点，不是经验证的亲和度预测；覆盖率高不等于靶点选择性高。")
        path = json.loads((folder / "pathways.json").read_text(encoding="utf8"))
        st.dataframe([{"查询": "、".join(r["query"]), "策略": r["method"], "背景": r["background"],
                       "映射蛋白数": r["protein_count"], "显著通路数": r.get("significant_pathways_fdr005",0)} for r in path["results"]], hide_index=True, width="stretch")
        st.caption("STRING 12.0官方接口；主分析采用当前49靶点背景。全人类背景为探索性对照。同一查询的三种方法得到相同靶点，当前结果不能支持PPI消融的独立富集优势。")
    else:
        st.caption("研究用途：对公开证据候选执行删除、增补、交换，检查相同目标是否改善。结果不作为处方。")
        selected = st.multiselect("集合优化症状", service.graph.symptom_names,
                                  default=[s for s in ["口渴", "乏力", "尿频"] if s in service.graph.symptom_names])
        excluded = st.multiselect("集合优化排除中药", service.graph.herb_names)
        if st.button("计算集合优化对照", type="primary"):
            from dmease.experiment_metrics import rollout, reference_score
            from dmease.research_policy import refine_selection
            task = {"symptoms": dict.fromkeys(selected, 1.), "forbidden": excluded}
            actions,_ = rollout(service.evidence, task)
            improved,diagnostic = refine_selection(service.evidence, task, actions)
            st.dataframe([{"阶段": label, "候选集合": "、".join(service.evidence.herbs[i]["name"] for i in indices),
                           "参考目标": reference_score(service.evidence, task, indices)["objective"], "数量": len(indices)}
                          for label,indices in [("证据增益",actions),("集合优化后",improved)]], hide_index=True, width="stretch")
            st.write(f"已比较{diagnostic['evaluated_subsets']}个集合，完成{diagnostic['rounds']}轮改进。")
    report = folder / "report.html"
    if report.exists():
        st.download_button("下载第二轮完整研究报告", report.read_bytes(), "DMease_第二轮研究报告.html", "text/html")
    st.download_button("下载第二轮策略结果", json_export(summary), "dmease_research_v2.json", "application/json")


def research_v3_evaluation(service):
    folder = ROOT / "experiments/v3"
    if not all((folder / name).exists() for name in ("affinity.json", "extraction.json", "syndrome_diagnostics.json")):
        st.info("第三轮实验结果尚未全部生成。")
        return
    section = st.selectbox("第三轮研究项目", ["实测亲和度", "本地模型抽取", "跨数据集诊断", "少样本改进"])
    if section == "实测亲和度":
        data = json.loads((folder / "affinity.json").read_text(encoding="utf8"))
        def method_name(configuration):
            if configuration["method"] == "mean":
                return "训练均值"
            if configuration["method"] == "ridge":
                return f"岭回归（α={configuration['alpha']}）"
            return f"随机森林（叶节点≥{configuration['min_samples_leaf']}）"
        st.caption("ChEMBL人源单蛋白Ki/Kd实测数据；按分子骨架隔离训练、验证和测试，各终点分别建模。")
        st.dataframe([{"终点": name, "训练化合物": r["counts"]["train"], "验证化合物": r["counts"]["validation"],
                       "测试化合物": r["counts"]["test"], "状态": "已评价" if r["status"] == "evaluated" else "样本量不足"}
                      for name,r in data["strata"].items()], hide_index=True, width="stretch")
        for name,r in data["strata"].items():
            if r["status"] != "evaluated":
                continue
            chosen = r["selection"]["selected_index"]
            st.write(f"{name}：验证选中模型的测试MAE为{r['test'][chosen]['MAE_pK']:.3f} pK，训练均值基线为{r['test'][0]['MAE_pK']:.3f} pK。")
            st.dataframe([{"模型": method_name(c["config"]), "验证MAE": c["validation"]["MAE_pK"],
                           "测试MAE": t["MAE_pK"], "测试R²": t["R2"], "验证选中": i == chosen}
                          for i,(c,t) in enumerate(zip(r["selection"]["candidates"],r["test"]))], hide_index=True, width="stretch")
        st.info("这是化合物结合强度预测，尚未与中药成分完成可靠结构连接，不能解释为整味中药的疗效或结合概率。修复前后结果均在报告中保留。")
    elif section == "本地模型抽取":
        data = json.loads((folder / "extraction.json").read_text(encoding="utf8"))
        if data["evidence_fingerprint"] != service.evidence.fingerprint:
            st.warning("本机词表与报告版本不同，以下指标仅对应报告的固定版本。")
        st.caption("固定版本Qwen2.5-1.5B-Instruct实际逐条推理；公开200条记录，输入仅为病例文本，输出要求引用原文。")
        labels = {"rules": "现有规则", "local_llm": "本地语言模型"}
        st.dataframe([{"方法": labels[m], "记录数": s["canonical_all"]["records"], "精确率": s["canonical_all"]["precision"],
                       "召回率": s["canonical_all"]["recall"], "F1": s["canonical_all"]["f1"],
                       "下游证候前三命中": s["downstream_exploratory"]["top3"]} for m,s in data["summary"].items()], hide_index=True, width="stretch")
        q = data["quality"]
        st.write(f"JSON格式失败{q['invalid_json']}条，输出达到长度上限{q['output_limit']}条；单条生成耗时中位数{q['median_seconds']:.1f}秒。失败记录计入总分母。")
        st.warning("参考症状和规则对照使用同一套24症状词典，投影标签不完整且会偏向规则方法。这是接口兼容性研究，不能作为公平的医学抽取排名或临床诊断准确率。当前辅助分析流程仍使用原验证路径。")
        audit_file = folder / "extraction_format_audit.json"
        if audit_file.exists():
            audit = json.loads(audit_file.read_text(encoding="utf8"))
            st.write(f"格式适配对照：只处理完整对象后的多余引号或外围代码框，JSON失败从{audit['strict_json_failures']}条变为{audit['remaining_json_failures']}条；同一批输出的投影F1为{audit['summary']['f1']:.3f}。")
            st.caption("这是观察格式问题后的工程处理，不是新的独立模型验证；未补写截断内容。")
        with st.expander("查看每种症状的抽取统计"):
            st.dataframe([{"方法": labels[m], "症状": name, "参考阳性": r["gold_count"], "正确抽取": r["tp"],
                           "多抽": r["fp"], "漏抽": r["fn"], "F1": r["f1"]} for m,items in data["per_concept"].items() for name,r in sorted(items.items())], hide_index=True, width="stretch")
    elif section == "跨数据集诊断":
        data = json.loads((folder / "syndrome_diagnostics.json").read_text(encoding="utf8"))
        t = data["transfer"]
        st.write(f"外部{t['records']}条病例中，仅{t['records_with_any_known_gold']}条具有分类器认识的证候标签；固定标签表下首候选命中率上限为{t['top1_known_label_ceiling']:.1%}，实际为{t['overall_top1']:.1%}。")
        st.dataframe([{"每类训练样本数": r["training_frequency"], "证候数": r["labels"], "测试记录数": r["test_records"],
                       "准确率": r["top1_accuracy"], "前三命中率": r["top3_hit"], "宏平均F1": r["macro_f1"]}
                      for r in data["frequency_bands"]], hide_index=True, width="stretch")
        st.dataframe([{"数据": name, "文本长度中位数": r["median_chars"], "字符特征覆盖率": r["mean_ngram_in_vocabulary"]}
                      for name,r in data["feature_coverage"].items()], hide_index=True, width="stretch")
        st.caption("复用上一轮冻结分类器；描述标签覆盖、写法差异和少样本问题，没有根据测试结果重新训练或选模。")
    else:
        if not (folder / "imbalance.json").exists():
            st.info("类别加权实验尚未完成。")
            return
        data = json.loads((folder / "imbalance.json").read_text(encoding="utf8"))
        labels = {"frozen_unweighted_v2": "原未加权基线", "validation_selected": "验证选中方案"}
        st.caption("训练类别频次决定样本权重；固定6个候选与原基线，按验证集宏平均F1选模。")
        st.dataframe([{"方法": labels[m], "范围": group, "记录数": r[group]["records"], "准确率": r[group]["accuracy"],
                       "宏平均F1": r[group]["macro_f1_148_train_labels"], "Kappa": r[group]["kappa"]}
                      for m,r in data["results"].items() for group in ("official_test", "unseen_text_test")], hide_index=True, width="stretch")
        st.dataframe([{"方法": labels[m], "每类训练样本数": b["training_frequency"], "测试记录": b["test_records"],
                       "准确率": b["accuracy"], "宏平均F1": b["macro_f1"]}
                      for m,r in data["results"].items() for b in r["frequency_bands"]], hide_index=True, width="stretch")
        st.warning("这是一项复用公开测试集的探索性改进，不是新的独立临床验证；收益属于类别加权文本分类，不能归为KAN/PPO收益。")
    report = folder / "report.html"
    if report.exists():
        st.download_button("下载第三轮完整研究报告", report.read_bytes(), "DMease_第三轮研究报告.html", "text/html")


def research(service):
    st.subheader("研究复现")
    st.caption("选择公开症状检索组合，使用与辅助分析相同的证据、约束和序贯策略；这里不是临床病例评价。")
    workspace = st.radio("研究工作区", ["序贯推理", "实验评估", "改进与外部评价", "亲和度与模型抽取", "结构化抽取与约束规划"], horizontal=True)
    if workspace == "结构化抽取与约束规划":
        from dmease.research_v4_ui import show
        show(service, ROOT)
        return
    if workspace == "亲和度与模型抽取":
        research_v3_evaluation(service)
        return
    if workspace == "改进与外部评价":
        research_v2_evaluation(service)
        return
    if workspace == "实验评估":
        experimental_evaluation(service)
        return
    defaults = [x for x in ["口渴", "乏力", "尿频"] if x in service.graph.symptom_names]
    selected = st.multiselect("选择研究症状", service.graph.symptom_names, default=defaults)
    forbidden = st.multiselect("研究排除中药", service.graph.herb_names)
    strategy = st.selectbox("研究策略", ["默认（依据验证集表现）", "公开证据增益", "KAN/PPO对照"])
    request_key = (tuple(selected), tuple(forbidden), service.evidence.signature, strategy)
    if st.button("运行序贯推理", type="primary"):
        result = service.analyze(PatientProfile(name="公开证据检索组合", forbidden_herbs=forbidden), selected,
                                 strategy={"默认（依据验证集表现）": None, "公开证据增益": "greedy", "KAN/PPO对照": "kan"}[strategy])
        st.session_state["research_result"] = (request_key, result)
    stored = st.session_state.get("research_result")
    if not stored or not isinstance(stored, tuple) or stored[0] != request_key:
        st.info("选择症状后点击运行。修改查询条件将要求重新运行。")
        return
    result = stored[1]
    trace = result.metadata["trace"]
    st.caption("实际策略：" + trace["mode"])
    if trace.get("strategy_notice"):
        st.warning(trace["strategy_notice"])
    view = st.radio("研究结果", ["推理摘要", "选择轨迹", "关联矩阵", "离线评价"], horizontal=True)
    if view == "推理摘要":
        a,b,c = st.columns(3)
        a.metric("症状证据覆盖", f"{trace['metrics']['symptom_evidence_coverage']:.1%}")
        b.metric("靶点证据覆盖", f"{trace['metrics']['target_evidence_coverage']:.1%}")
        c.metric("入选中药", len(result.recommendations))
        st.write("停止原因：" + trace["stop_reason"])
        st.caption("无相关靶点证据时靶点覆盖记为0，不表示疗效为0。证据覆盖不表示症状改善。")
        st.dataframe([{"中药": x.name, "证据增益": x.score} for x in result.recommendations], hide_index=True, width="stretch")
    elif view == "选择轨迹":
        rows = [{"步骤": x["step"], "中药": x["herb"], "证据增益": x["marginal_evidence_gain"],
                 "新增靶点": "、".join(x["new_targets"]), "未覆盖证据量": sum(x["residual_evidence"].values())} for x in trace["steps"]]
        st.dataframe(rows, hide_index=True, width="stretch")
        if rows:
            st.line_chart(pd.DataFrame(rows).set_index("步骤")[["未覆盖证据量"]], height=220)
    elif view == "关联矩阵":
        attribution = pd.DataFrame({x["herb"]: x["gradient_attribution"] for x in trace["steps"]}).fillna(0)
        st.write("策略对当前未覆盖症状状态的梯度")
        st.dataframe(attribution, width="stretch")
        st.caption("梯度属于模型局部敏感度，不是医学因果关系。无匹配KAN模型时使用证据增益策略，梯度为空。")
    else:
        experimental_evaluation(service)
        path = ROOT / "models/public_evidence_evaluation.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf8"))
            if report["metadata"]["signature"] != service.evidence.signature:
                st.warning("下列报告属于其他证据版本。")
            st.dataframe([{"策略": name, "测试查询数": item["queries"], **item["mean"]}
                          for name, item in report["evaluation"].items()], hide_index=True, width="stretch")
            st.caption("按查询组合划分训练、验证、测试，底层知识图谱共享；指标是同一公开图谱上的覆盖与约束检查，不代表独立临床验证。")
        else:
            st.info("当前部署尚无训练评价报告。")
    st.download_button("导出研究结果 JSON", json_export(result.to_dict()), "dmease_research.json", "application/json")
    st.download_button("导出选择轨迹 CSV", csv_export(trace["steps"]), "dmease_trace.csv", "text/csv")


def settings(service, auth, owner):
    st.subheader("系统与账号")
    section = st.radio("设置项目", ["运行状态", "修改密码", "使用说明"], horizontal=True)
    if section == "运行状态":
        st.write(f"**软件名称**　{SOFTWARE_NAME}　{VERSION}")
        st.write(f"**当前排序模式**　{service.recommender.mode}")
        st.info(service.recommender.load_message)
        st.write(f"**知识版本**　{service.graph.summary()['version']}")
        st.write(f"**患者记录**　{service.database.count(owner)}条（当前账号）")
        st.write("**会话策略**　30分钟无操作后，下次交互要求重新登录。")
        if service.recommender.model_metadata:
            meta = service.recommender.model_metadata
            st.write(f"**训练记录**　非临床图谱派生任务；{meta.get('iterations', '未记录')}轮，随机种子{meta.get('seed', '未记录')}。")
            st.caption("模型指纹：" + meta.get("checkpoint_sha256", "未记录"))
            with st.expander("查看模型完整元数据"):
                st.json(service.recommender.model_metadata)
        st.caption("部署者可通过DMEASE_CHECKPOINT指定训练检查点；更换检查点后重新加载页面即可识别更新。")
    elif section == "修改密码":
        with st.form("change_password", clear_on_submit=True):
            old = st.text_input("当前密码", type="password", max_chars=128)
            new = st.text_input("新密码", type="password", max_chars=128)
            repeat = st.text_input("再次输入新密码", type="password", max_chars=128)
            st.caption("密码12—128位，包含英文字母和数字；修改成功后需要重新登录。")
            submit = st.form_submit_button("更新密码", type="primary")
        if submit:
            try:
                if new != repeat:
                    raise ValueError("两次输入的新密码不一致。")
                auth.change_password(owner, old, new)
                clear_session("密码已更新，请使用新密码登录。")
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    else:
        st.markdown("""### 使用范围
系统提供症状解析、证候判断、中药候选排序、约束解释、知识查询和记录管理。
正式流程使用从官方入口取得并校验的公开数据库关系。训练数据是公开证据检索任务，不是患者病例。结果不构成处方或剂量建议。
### 记录与导出
分析结果在核对后保存；复用旧资料会创建新分析，保留旧记录。JSON保存完整结构，CSV用于表格分析，HTML报告可打印。
### 部署维护
账号密码以随机盐和PBKDF2摘要保存。部署者应备份运行数据目录，并在共享网络部署时配置HTTPS与访问控制。
### 模型运行
公开证据KAN/PPO训练入口为scripts/train_evidence_policy.py。辅助分析与研究页使用同一个序贯环境，每一步更新证据覆盖、已选药材与约束。缺少匹配模型时使用同一公开证据的增益策略，不使用构造数据。体质与病史仅记录，尚未完成经验证的个体化学习。""")


auth = load_auth(str(DATA))
owner = authenticate(auth)
try:
    service = load_service(str(DATA), str(CHECKPOINT), CHECKPOINT.stat().st_mtime_ns if CHECKPOINT.exists() else 0,
                           str(BUNDLE), BUNDLE.stat().st_mtime_ns if BUNDLE.exists() else 0)
except (ValueError, OSError, RuntimeError, KeyError) as error:
    st.error("公开证据尚未初始化或校验失败。请按项目说明下载公开数据并构建证据，再启动分析；系统不会改用构造知识库。")
    st.caption(type(error).__name__)
    st.stop()
if "pending_navigation" in st.session_state:
    st.session_state["navigation"] = st.session_state.pop("pending_navigation")
with st.sidebar:
    st.title("DMease")
    st.caption("中医糖尿病智能辅助诊疗系统")
    st.caption(f"{VERSION} · 当前账号：{owner}")
    page = st.radio("功能导航", PAGES, key="navigation")
    st.divider()
    st.caption("数据：SymMap / STRING公开关联")
    st.button("退出登录", on_click=clear_session, args=("已退出登录。",), width="stretch")
st.title(SOFTWARE_NAME)
st.caption(f"{VERSION}  /  {page}")
if page == "首页": home(service, owner)
elif page == "辅助分析": analysis(service, owner)
elif page == "患者记录": records(service, owner)
elif page == "知识图谱": graph_page(service)
elif page == "研究复现": research(service)
else: settings(service, auth, owner)

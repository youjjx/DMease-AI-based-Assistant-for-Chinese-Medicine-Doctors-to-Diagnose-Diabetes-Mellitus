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


def research(service):
    st.subheader("研究复现")
    st.caption("选择公开症状检索组合，使用与辅助分析相同的证据、约束和序贯策略；这里不是临床病例评价。")
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

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dmease import DMeaseService
from dmease.types import PatientProfile


ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="DMease 中医糖尿病辅助诊疗", page_icon="🌿", layout="wide")


@st.cache_resource
def load_service() -> DMeaseService:
    return DMeaseService(
        graph_path=ROOT / "data" / "knowledge_graph.json",
        database_path=ROOT / "data" / "patients.db",
        checkpoint_path=ROOT / "checkpoints" / "kan_ppo.pt",
    )


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]


def render_analysis(result) -> None:
    st.subheader("证候判断")
    if not any(item.present for item in result.parsed_symptoms):
        st.warning("未识别到阳性症状，请补充主诉或从症状列表中勾选。")
        return

    cols = st.columns(len(result.syndromes))
    for column, syndrome in zip(cols, result.syndromes):
        with column:
            st.metric(syndrome.name, f"{syndrome.confidence:.1%}", f"证据分 {syndrome.score:.2f}")
            st.caption("匹配：" + ("、".join(syndrome.matched_symptoms) or "无"))

    with st.expander("查看症状解析与证候依据", expanded=True):
        symptom_rows = [
            {
                "标准症状": item.canonical,
                "状态": "阳性" if item.present else "否定",
                "强度": item.severity,
                "原文证据": item.evidence,
            }
            for item in result.parsed_symptoms
        ]
        st.dataframe(symptom_rows, use_container_width=True, hide_index=True)
        for syndrome in result.syndromes:
            st.markdown(f"**{syndrome.name}**：{syndrome.rationale}")
            if syndrome.missing_key_symptoms:
                st.caption("尚未出现的关键线索：" + "、".join(syndrome.missing_key_symptoms))

    st.subheader("候选中药排序")
    if result.recommendations:
        chart = pd.DataFrame(
            {"中药": [item.name for item in result.recommendations], "综合得分": [item.score for item in result.recommendations]}
        ).set_index("中药")
        st.bar_chart(chart)
        for item in result.recommendations:
            with st.expander(f"#{item.rank} {item.name}　得分 {item.score:.3f}"):
                st.write("功效标签：" + "、".join(item.actions))
                st.write("关联证候：" + "、".join(item.matched_syndromes))
                for path in item.evidence_paths:
                    st.code(path, language=None)
                for warning in item.warnings:
                    st.caption("提示：" + warning)
    else:
        st.info("当前输入和安全约束下没有可展示的候选中药。")

    if result.excluded_herbs:
        with st.expander(f"安全约束已排除 {len(result.excluded_herbs)} 味中药"):
            for herb, reasons in result.excluded_herbs.items():
                st.write(f"- {herb}：{'；'.join(reasons)}")

    st.error("\n\n".join(result.safety_notes))
    st.download_button(
        "导出本次分析 JSON",
        data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        file_name="dmease_analysis.json",
        mime="application/json",
    )


service = load_service()
st.title("🌿 DMease 中医糖尿病辅助诊疗系统")
st.caption("知识图谱 + 症状解析 + 证候推理 + KAN/PPO 候选中药排序｜完整研究实现")
st.warning("本系统不能替代医生，不提供处方与剂量，输出不得直接用于临床诊疗。")

tab_analysis, tab_records, tab_graph, tab_about = st.tabs(["辅助分析", "患者记录", "知识图谱", "系统说明"])

with tab_analysis:
    with st.form("patient_form"):
        left, right = st.columns(2)
        with left:
            name = st.text_input("患者标识", value="匿名患者", help="建议使用匿名编号")
            age = st.number_input("年龄", min_value=1, max_value=120, value=50)
            sex = st.selectbox("性别", ["未说明", "男", "女"])
            constitution = st.selectbox(
                "体质",
                ["平和质", "气虚质", "阴虚质", "阳虚质", "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"],
            )
            pregnant = st.checkbox("妊娠期（用于安全约束）", disabled=sex == "男")
        with right:
            allergies = st.text_input("已知过敏（逗号分隔）")
            conditions = st.text_input("合并情况（逗号分隔）", placeholder="如：慢性腹泻、高血压未控制")
            medications = st.text_input("当前用药（逗号分隔）", placeholder="如：华法林")
            notes = st.text_area("补充说明")
        complaint = st.text_area(
            "主诉与症状描述",
            height=130,
            placeholder="示例：近三个月口渴多饮、容易疲劳，口干明显，夜间小便多，无肢体麻木。",
        )
        selected = st.multiselect("补充勾选症状", service.graph.symptom_names)
        submitted = st.form_submit_button("开始辅助分析", type="primary", use_container_width=True)

    if submitted:
        profile = PatientProfile(
            name=name or "匿名患者",
            age=int(age),
            sex=sex,
            constitution=constitution,
            complaint=complaint,
            allergies=split_values(allergies),
            conditions=split_values(conditions),
            pregnant=bool(pregnant and sex != "男"),
            medications=split_values(medications),
            notes=notes,
        )
        st.session_state["analysis_result"] = service.analyze(profile, selected)

    result = st.session_state.get("analysis_result")
    if result:
        render_analysis(result)
        if st.button("保存到本地患者数据库"):
            record_id = service.database.save(result)
            st.success(f"已保存为记录 #{record_id}")

with tab_records:
    st.subheader("本地患者记录")
    records = service.database.list()
    if not records:
        st.info("暂无保存记录。完成一次辅助分析后可手动保存。")
    else:
        st.dataframe(records, use_container_width=True, hide_index=True)
        record_ids = [item["id"] for item in records]
        chosen = st.selectbox("查看记录", record_ids, format_func=lambda value: f"记录 #{value}")
        payload = service.database.get(int(chosen))
        st.json(payload, expanded=False)

with tab_graph:
    summary = service.graph.summary()
    cols = st.columns(4)
    cols[0].metric("症状实体", summary["symptoms"])
    cols[1].metric("证候实体", summary["syndromes"])
    cols[2].metric("中药实体", summary["herbs"])
    cols[3].metric("成分/靶点/通路", summary["compounds"] + summary["targets"] + summary["pathways"])
    st.caption(f"当前知识图谱共 {summary['relations']} 条关系，版本 {summary['version']}。")
    syndrome_name = st.selectbox("选择证候查看关联", service.graph.syndrome_names)
    syndrome = service.graph.syndrome(syndrome_name)
    st.write(syndrome["description"])
    st.dataframe(
        [{"症状": name, "关联权重": weight} for name, weight in syndrome["symptoms"].items()],
        use_container_width=True,
        hide_index=True,
    )
    mechanism_herb = st.selectbox("查看成分—靶点—通路路径", service.graph.herb_names)
    mechanism_paths = service.graph.mechanism_paths(mechanism_herb)
    if mechanism_paths:
        for path in mechanism_paths:
            st.code(path, language=None)
    else:
        st.caption("知识库尚未为该中药配置机制路径。")
    st.dataframe(
        [
            {"中药": item["herb"]["name"], "关联权重": item["weight"], "功效": "、".join(item["herb"]["actions"])}
            for item in service.graph.herbs_for_syndrome(syndrome_name)
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_about:
    st.markdown(
        """
### 系统实现范围

系统将自由文本主诉归一化为症状实体，通过可解释的加权图谱推断候选证候，再由图谱先验与可选的 KAN/PPO 策略联合生成中药候选排序。硬约束会排除过敏、妊娠、特定合并情况和已知相互作用项。

### 数据边界

系统内置可直接运行的非临床样例知识数据，并完整实现论文所述的核心软件流程。数据接口可接入经授权、完成脱敏和质量控制的临床数据；患者记录默认保存在本机 SQLite 文件中。

### 使用原则

输出用于研究、教学和工程验证。临床部署前须完成伦理、隐私、安全、药学审查和前瞻性临床验证。
"""
    )

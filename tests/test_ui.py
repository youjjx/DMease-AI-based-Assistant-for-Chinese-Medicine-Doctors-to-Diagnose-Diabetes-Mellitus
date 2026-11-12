from pathlib import Path

from streamlit.testing.v1 import AppTest

from dmease.auth import AuthStore

APP = Path(__file__).parents[1] / "app.py"


def widget(elements, label):
    return next(x for x in elements if x.label == label)


def test_login_analysis_save_reuse_and_logout(tmp_path, monkeypatch, evidence_path):
    monkeypatch.setenv("DMEASE_EVIDENCE_BUNDLE", str(evidence_path))
    monkeypatch.setenv("DMEASE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DMEASE_CHECKPOINT", str(tmp_path / "absent.pt"))
    AuthStore(tmp_path / "accounts.db").bootstrap("operator", "Test-password1234")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert not app.sidebar.radio
    widget(app.text_input, "账号").set_value("operator")
    widget(app.text_input, "密码").set_value("wrong")
    widget(app.button, "登录").click().run()
    assert app.error and not app.sidebar.radio
    widget(app.text_input, "账号").set_value("operator")
    widget(app.text_input, "密码").set_value("Test-password1234")
    widget(app.button, "登录").click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "首页"
    widget(app.button, "新建辅助分析").click().run()
    widget(app.text_input, "患者标识 *").set_value("P001")
    widget(app.text_input, "已知过敏").set_value("黄芪")
    widget(app.text_area, "主诉与症状描述 *").set_value("口渴多饮、容易疲劳，口干明显，夜间小便多，无肢体麻木。")
    widget(app.button, "开始辅助分析").click().run()
    assert not app.exception
    assert app.session_state["analysis_step"] == "2 症状与证候"
    widget(app.button, "下一步：候选与解释").click().run()
    widget(app.button, "下一步：约束与保存").click().run()
    assert widget(app.button, "保存本次分析").disabled
    widget(app.checkbox, "已核对本次输入、约束提示与结果使用范围").check().run()
    widget(app.button, "保存本次分析").click().run()
    assert not app.exception
    assert app.session_state["saved_id"] == 1
    assert widget(app.button, "保存本次分析").disabled
    widget(app.button, "前往患者记录").click().run()
    widget(app.button, "复用资料并重新分析").click().run()
    assert not app.exception
    assert widget(app.text_input, "患者标识 *").value == "P001"
    widget(app.text_area, "主诉与症状描述 *").set_value("")
    widget(app.button, "开始辅助分析").click().run()
    assert app.error
    assert "analysis_result" not in app.session_state
    widget(app.button, "退出登录").click().run()
    assert not app.exception and not app.sidebar.radio
    assert "draft" not in app.session_state and "account" not in app.session_state


def test_research_and_knowledge_pages(tmp_path, monkeypatch, evidence_path):
    monkeypatch.setenv("DMEASE_EVIDENCE_BUNDLE", str(evidence_path))
    monkeypatch.setenv("DMEASE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DMEASE_CHECKPOINT", str(tmp_path / "absent.pt"))
    AuthStore(tmp_path / "accounts.db").bootstrap("operator", "Test-password1234")
    app = AppTest.from_file(str(APP), default_timeout=60)
    import time
    for key, value in dict(account="operator", credential_revision=1, last_activity=time.time()).items():
        app.session_state[key] = value
    app.run()
    app.sidebar.radio[0].set_value("知识图谱").run()
    widget(app.radio, "图谱视图").set_value("知识子图").run()
    assert not app.exception
    widget(app.radio, "图谱视图").set_value("机制路径").run()
    assert not app.exception
    widget(app.radio, "图谱视图").set_value("公开数据检索").run()
    assert not app.exception
    if any(x.label == "检索名称或原始标识" for x in app.text_input):
        widget(app.text_input, "检索名称或原始标识").set_value("AKT1").run()
        assert not app.exception
        widget(app.selectbox, "公开数据来源").set_value("TCMSP").run()
        widget(app.text_input, "检索名称或原始标识").set_value("黄芪").run()
        assert not app.exception
    app.sidebar.radio[0].set_value("研究复现").run()
    widget(app.button, "运行序贯推理").click().run()
    assert not app.exception
    widget(app.radio, "研究结果").set_value("选择轨迹").run()
    assert not app.exception
    widget(app.radio, "研究结果").set_value("关联矩阵").run()
    assert not app.exception
    widget(app.radio, "研究工作区").set_value("实验评估").run()
    assert not app.exception
    assert any("证据版本不同" in x.value for x in app.warning)
    for section in ("约束压力测试", "参数敏感性", "证据缺失", "精确最优解对照", "文本解析回归", "模型与消融"):
        widget(app.selectbox, "实验项目").set_value(section).run()
        assert not app.exception
    widget(app.radio, "研究工作区").set_value("序贯推理").run()
    assert not app.exception
    widget(app.radio, "研究工作区").set_value("改进与外部评价").run()
    assert not app.exception
    for section in ("真实病例辨证", "通路与靶点一致性", "单次集合优化"):
        widget(app.selectbox, "改进研究项目").set_value(section).run()
        assert not app.exception
    widget(app.multiselect, "集合优化排除中药").set_value(["黄芪"]).run()
    widget(app.button, "计算集合优化对照").click().run()
    assert not app.exception
    comparison = app.dataframe[-1].value
    assert all("黄芪" not in value for value in comparison["候选集合"])
    assert comparison.iloc[1]["参考目标"] >= comparison.iloc[0]["参考目标"] - 1e-7
    widget(app.radio, "研究工作区").set_value("亲和度与模型抽取").run()
    assert not app.exception
    if (APP.parent / "experiments/v3/extraction.json").exists():
        for section in ("实测亲和度", "本地模型抽取", "跨数据集诊断", "少样本改进"):
            widget(app.selectbox, "第三轮研究项目").set_value(section).run()
            assert not app.exception and app.dataframe
    widget(app.radio, "研究工作区").set_value("结构化抽取与约束规划").run()
    assert not app.exception
    for section in ("抽取验证", "规划对照", "外部辨证与亲和度", "原文抽取核对", "约束规划试算"):
        widget(app.selectbox, "第四轮研究项目").set_value(section).run()
        assert not app.exception
    if (APP.parent / "experiments/v4/planning_training/residual_kan_42.json").exists():
        widget(app.button, "比较规划结果").click().run()
        assert not app.exception
        assert any("证据图与模型训练版本不一致" in message.value for message in app.error)
    app.session_state["last_activity"] = 0
    app.run()
    assert not app.sidebar.radio and "account" not in app.session_state

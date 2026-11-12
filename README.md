# DMease中医糖尿病智能辅助诊疗系统 V1.0

DMease is a complete research implementation for the paper **"DMease: AI-based Assistant Tool for Traditional Chinese Medicine Doctors in Diagnosing and Treating Diabetes Mellitus"**.

The repository has been rebuilt around the method described in the paper:

1. Multi-source TCM text is normalized and converted into structured triples.
2. Triples are stored as a symptom-target-herb knowledge graph.
3. A PatientDB provides patient-ID based records.
4. The system predicts a TCM syndrome from symptoms.
5. A sequential single-herb policy ranks and selects herbs under compatibility constraints.
6. The output includes target coverage, marginal symptom relief, and provenance traces.

The repository includes directly runnable non-clinical sample data. The software is for research and engineering validation; clinical deployment requires authorized data, governance review, and clinician oversight.

## 中文操作入口

安装 Python 3.10+ 后，在项目根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m streamlit run app.py
```

Windows 也可双击 `start_app.bat`。默认仅监听本机 `127.0.0.1:8501`。首次部署应先在受控环境完成账号初始化，再由部署者配置网络访问。浏览器进入后创建账号（3—32位字母、数字或下划线）和密码（12—128位，包含字母与数字）。仓库不含默认账号或密码。

登录后的功能导航：

| 功能 | 使用方式 |
| --- | --- |
| 首页 | 查看当前账号的保存数量、知识规模、排序模式和最近记录 |
| 辅助分析 | 资料录入 → 症状与证候 → 候选与解释 → 约束与保存 |
| 患者记录 | 按患者标识、主诉和证候筛选；查看、导出、复用或确认删除记录 |
| 知识图谱 | 查询证候关联、中药机制路径与可视化知识子图 |
| 研究复现 | 执行已有JSONL样例的靶点检索与确定性序贯推理，查看轨迹和贡献矩阵 |
| 系统与账号 | 查看实际模型状态和知识版本，修改密码、阅读使用说明 |

输入草稿可保留在当前会话中；切换页面前点击“保存输入草稿”。刷新、关闭浏览器或退出登录会结束当前会话，已保存数据库记录不受影响。修改输入后重新分析；从历史记录复用资料会创建关联的新记录，原始结果保留。同一分析重复保存不会插入重复记录。保存前需勾选核对提示。

结果支持完整JSON、候选CSV、记录列表CSV和HTML打印报告。HTML报告下载后在浏览器中打印或另存为PDF。CSV使用UTF-8 BOM并处理可能被表格软件识别为公式的文本。

### 账号与数据

密码以随机盐和PBKDF2-HMAC-SHA256摘要保存；连续5次失败暂停验证60秒。30分钟无操作后，下次交互要求重新登录。密码修改使其他旧会话在下次交互时失效。退出会清除当前会话的患者资料、结果和草稿。当前部署提供一个初始化操作账号；没有开放注册或多角色管理。

运行数据默认位于忽略上传的 `runtime/`，包括 `accounts.db` 和 `patients.db`。可通过 `DMEASE_DATA_DIR`指定独立目录。请由部署者定期备份该目录，关闭写入后复制两个数据库；恢复时保持账号数据库与患者数据库配套。旧版 `data/patients.db` 不会自动搬入新目录；其旧记录无账号归属，不会自动向登录账号开放。部署者应先备份，再按实际数据归属制定迁移方案。

### KAN/PPO模型

仓库包含 `models/nonclinical_kan_ppo.pt`，由本仓库训练程序使用非临床构造知识图谱训练生成。应用默认加载该检查点；若存在 `checkpoints/kan_ppo.pt` 则优先加载，环境变量 `DMEASE_CHECKPOINT`可覆盖路径。模型加载时核对实体顺序与图谱版本。损坏或不匹配时显示原因并使用图谱评分；运行状态和每次结果元数据记录实际模式与模型指纹。

默认混合得分为85%图谱评分和15%KAN策略概率，硬约束始终参与候选过滤。非临床训练权重可证明训练、加载和推理流程运行，不能作为临床效果证据。研究复现页采用独立的确定性序贯算法，其模型症状减量不表示实际治疗效果。


## Repository Structure

```text
configs/
  dmease.yaml                  # Main configuration
data/
  README.md                    # Expected data layout
  examples/                    # Non-clinical sample data for validation and tests
  knowledge_graph.json         # Structured symptom-syndrome-herb-mechanism graph
prompts/
  triplet_extraction_zh.md     # LLM triplet extraction prompt
src/dmease/
  schemas.py                   # Patient, triple, diagnosis, prescription schemas
  patient_db.py                # Lightweight PatientDB
  semantic_parser.py           # Surface-to-concept mapping and JSON triplet parser
  knowledge_graph.py           # Symptom-target-herb KG and PMI-style retrieval
  diagnosis.py                 # Syndrome diagnosis module
  constraints.py               # Classical and patient-specific hard constraints
  kan.py                       # KAN actor, critic, and edge-function layers
  ppo.py                       # Clipped PPO environment and trainer
  policy.py                    # Current greedy single-herb policy
  recommender.py               # Hybrid graph/KAN ranking and constraints
  symptom_parser.py            # Chinese symptom normalization and negation
  auth.py                      # Local account verification and credential lifecycle
  exports.py                   # JSON/CSV export and escaped printable HTML
  service.py                   # Interactive end-to-end service facade
  types.py                     # Interactive workflow data structures
  pipeline.py                  # End-to-end inference workflow
  cli.py                       # Command line interface
scripts/
  build_kg.py                  # Convert LLM JSON output to triples.jsonl
  infer_example.py             # Minimal inference example
app/
  streamlit_app.py             # Patient-ID workflow
app.py                         # Full Chinese Streamlit workflow
tests/
  test_pipeline.py             # Smoke tests for the example pipeline
```

## Installation

```bash
conda env create -f environment.yml
conda activate dmease
pip install -e .
```

The included non-clinical sample assets under `data/examples/` and `data/knowledge_graph.json` make both inference workflows directly reproducible.

## Quick Start

List example patients:

```bash
dmease --config configs/dmease.yaml patients
```

Run the paper-style sample case:

```bash
dmease --config configs/dmease.yaml infer --patient-id P2025-0001
```

Or run directly from source:

```bash
PYTHONPATH=src python scripts/infer_example.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"; python scripts/infer_example.py
```

## Data Expected by the Paper

The paper describes the following sources:

- SymMap: TCM symptoms and gene/protein targets.
- STRING v12.0: human PPI network with confidence filtering.
- TCMSP: herbs and chemical ingredients.
- Classical literature: 30 foundational TCM texts after OCR correction.
- Guidelines: 5 national TCM guidelines.
- De-identified EMR: 108,746 TCM outpatient records.

Keep licensed or private datasets outside Git. For local experiments, put source files under `data/raw/`, then generate:

```text
data/processed/patients.jsonl
data/processed/triples.jsonl
data/processed/herb_target_affinity.csv
```

The included `data/examples/` files are compact non-clinical sample assets aligned with the paper's case structure. They make the repository runnable out of the box for tests and engineering review.

## Knowledge Graph Triples

The graph uses the ten relation templates from the paper:

| ID | Template |
| --- | --- |
| R1 | Herb-affect by-Target |
| R2 | Symptom-associated with-Target |
| R3 | Symptom-indicates-Syndrome |
| R4 | Syndrome-treatedBy-Herb |
| R5 | Herb-contains-Compound |
| R6 | Compound-targets-Gene |
| R7 | Gene-associatedWith-Symptom |
| R8 | Herb-contraindicatedWith-Herb |
| R9 | Syndrome-hasStage-DiseaseStage |
| R10 | Gene-participatesIn-Pathway |

LLM output should be strict JSON. See `prompts/triplet_extraction_zh.md`.

## Inference Policy

The paper describes a KAN policy trained with PPO for sequential single-herb prescription.

This repository implements:

- KAN actor and critic networks with trainable edge-wise basis functions in `src/dmease/kan.py`.
- A clipped PPO environment, rollout collector, optimizer, and checkpoint writer in `src/dmease/ppo.py`.
- A deterministic sequential policy for reproducible PatientDB inference in `src/dmease/policy.py`.
- A graph/KAN hybrid ranking service with hard safety constraints and evidence paths in `src/dmease/recommender.py`.
- A `train.py` workflow that trains the KAN/PPO policy on graph-derived sequential herb-selection tasks.

The interactive workflow uses the JSON syndrome/herb graph with optional KAN probabilities and hard constraints. The research reproduction workflow separately uses JSONL triples, target affinity and a deterministic sequential policy. Their output structures are documented separately; graph-derived training is not clinical validation.

Train and save a KAN/PPO checkpoint:

```bash
python train.py --iterations 50 --episodes 8
```

## Streamlit UI

```bash
streamlit run app.py
```

The Chinese UI supports free-text symptom parsing, syndrome inference, constrained herb ranking, evidence paths, knowledge graph browsing, and persistent SQLite patient records. `app/streamlit_app.py` opens the same authenticated interface; the paper-style patient-ID workflow is available on the 研究复现 page.

## Validation

```bash
pytest
```

The tests verify both workflows: symptom negation, syndrome reasoning, evidence paths, safety constraints, SQLite persistence, target retrieval, sequential selection, and patient-specific allergy handling.

## Citation

Please cite the paper if this repository is useful for your work:

```bibtex
@inproceedings{jia2025dmease,
  title     = {DMease: AI-based Assistant Tool for Traditional Chinese Medicine Doctors in Diagnosing and Treating Diabetes Mellitus},
  author    = {Jia, Junxiang and Hao, Xinyi and Hu, Yizhong and Meng, Haoyang and Zhao, Cong and Guan, Jianfeng and Sun, Weiwei},
  booktitle = {IEEE BIBM},
  year      = {2025}
}
```

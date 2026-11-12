# Data Layout

第五轮新增发布范围的公开证据快照 `experiments/v5/planning_features.json`，以及经过身份过滤的 `molecular_bundle.json`。前者是实际模型输入和界面所需的关系子集；后者包含1,128条药材成分连接、816个PubChem结构、2条匹配实测及84条域内预测。它们不是完整原始数据库。可直接运行发布版本，重新采集仍按下述原始数据布局进行。

本轮新增结构来自PubChem实际CID查询；实测亲和度复用已核验的ChEMBL37快照。每条边保留来源标识，存在多个候选CID、分子量或元素式冲突的路径不进入推荐。原始数据库立体结构没有全部独立取得，预测适用域和误差带也不构成临床验证。详见 `experiments/v5/DATA_LICENSE.txt`、`identity_audit.json`、`affinity_evaluation.json` 和 `source_availability.json`。

The original paper describes three data sources:

- Classical TCM literature: 30 foundational books after OCR correction.
- Guidelines: 5 national TCM guidelines for diabetes, hypertension, and chronic kidney disease.
- De-identified EMR: 108,746 outpatient TCM records.

Keep licensed or private data out of Git and use the following layout for local research datasets:

```text
data/
  raw/
    classical_literature/
    guidelines/
    emr/
    symmap/
    string/
    tcmsp/
  processed/
    patients.jsonl
    triples.jsonl
    herb_target_affinity.csv
  examples/
    patients.jsonl
    triples.jsonl
    herb_target_affinity.csv
```

The `examples/` directory contains a compact paper-aligned non-clinical sample dataset for reproducible validation and tests.
It is not clinical evidence and must not be used for real medical decisions.

## 实际数据来源与边界

`knowledge_graph.json`、`examples/patients.jsonl`、`examples/triples.jsonl` 和 `examples/herb_target_affinity.csv` 是人工构造的非临床数据。亲和度数值不是实验测量或由公开数据库训练出的预测值。历史权重 `models/nonclinical_kan_ppo.pt` 由这些构造关系训练，正式入口已不再使用。新权重见 `models/public_model_manifest.json`。

`scripts/import_public_data.py` 新增独立的官方数据获取流程：

| 数据源 | 取得范围 | 来源与许可记录 |
| --- | --- | --- |
| STRING 12.0 | 完整人类蛋白信息及 links 文件；索引筛选 combined_score≥700，AB/BA合并 | https://version-12-0.string-db.org/cgi/download；CC BY 4.0；署名 STRING Consortium |
| SymMap 1.0/2.0 | 官方下载页的所有 XLSX 实体描述表、标识表，保留 Suppress 等原始字段 | http://www.symmap.org/download/；未在下载页识别到明确再分发许可，原始文件不随本仓库发布 |
| TCMSP 2.3 | herbs、ingredients、targets、diseases 四类完整公开浏览表，以及当前应用药材精确名称匹配的成分/靶点详情 | https://old.tcmsp-e.com/tcmsp.php；ODbL 1.0 / DbCL 1.0，原始文件与索引独立存放 |

引用 TCMSP 时请注明 Ru et al., *TCMSP: a database of systems pharmacology for drug discovery from herbal medicines*, Journal of Cheminformatics 6, 13 (2014)。SymMap 请按其官方网站参考文献引用，版本以下载文件为准。

`public_registry.json` 是实际导入报告，而非原始数据。每个文件记录 URL、取得时间、字节数、SHA-256；SHA-256 用于识别本次取得的文件与检测缓存变更，不代表来源网站对内容作了签名。SymMap 在本次核验时 HTTPS 失败，使用其官方 HTTP 下载入口；这一传输限制保留在报告中。

下载缓存与索引不包含患者资料。SymMap 各版本、各表和各数据源分别保存，不把交叉引用行或同一实体的多个版本重复算成独立医学实体。目录完整不等于关系完整，文件中含被标记停用的条目时保留标记，不自动选入推荐。

STRING 的综合关联分数不是结合亲和度，功能关联也不一定是物理相互作用。TCMSP 的关联包含预测信息，原始 validated、SVM_score、RF_score 保存在关系来源中；缺失值不会伪造为已验证。精确名称未匹配的药材列在报告中，不擅自合并生地黄/地黄等可能存在炮制差异的名称。

正式分析现在使用 `build_evidence_bundle.py` 取得的SymMap关系与STRING关联。语义指纹排除下载时间，关系与显著性字段保持可追溯。TCMSP用于独立路径核查；生产推荐尚未接入实测亲和度预测，完整实体对齐与原论文同病例评价仍未完成。第三轮独立亲和度实验见下文。本次导入不包含原论文的108,746条门诊记录或50例专家标注，也不构成论文实验复现。

## 第二轮公开病例与通路研究

新研究独立使用以下来源，未混入生产图谱，病例原文与训练文本模型不随Git分发：

| 来源 | 实际取得与用途 | 许可与引用 |
| --- | --- | --- |
| TCMEval-SDT | 作者仓库的Train_TCM_Data_v1.json，200条带答案记录；冻结图谱的外部零样本审计及探索性迁移 | 作者说明数据CC BY 4.0；Wang等，Scientific Data 12, 437 (2025)，https://doi.org/10.1038/s41597-025-04772-9；数据DOI https://doi.org/10.6084/m9.figshare.27184596.v4 |
| TCM-SD | 作者仓库固定提交72a89518392ec28f5dc1c8ed252a6c127ca2145b；train/dev/test为43,180/5,486/5,486条；监督文本分类对照 | 数据CC BY-NC-SA 4.0，作者README将其与MIT代码许可区分；Ren等，CCL 2022，https://aclanthology.org/2022.ccl-1.80/；https://github.com/Borororo/ZY-BERT |
| STRING 12.0 enrichment | 3个固定查询×3种策略×2种背景；缓存实际API响应和指纹，报告实际返回的KEGG/Reactome条目 | STRING官方 https://string-db.org/help/api/；请求记录见experiments/v2/pathways.json |

更正：此前仅检查病例JSON，误判TCMEval-SDT验证/测试答案未公开。作者实际将答案单独放在`evaluation/TCMEval-SDT/Results`，`evaluate.py`也将其作为标准答案。第四轮已按病例ID对齐50条验证和50条测试记录，固定作者提交`12473b50ce94ad2fd22b1e984b826b0a618c4407`；来源校验见`experiments/v4/extraction_sources.json`。旧轮200条训练分区的数值仍保留，其身份不改为测试分区。

TCM-SD只把主诉、病情描述、检查文本作为输入；标识、疾病名称和证候字段不作为特征。重复文本与源user_id重叠计数保留在报告中，后者含义未核实，尚不能保证患者级独立划分。

公开病例的辨证参考标签不是中药疗效或亲和度标签，也不能补出原论文的特定糖尿病病例。生产入口的证候模块仍使用公开图谱，研究分类器不自动替换临床工作流。

## 第三轮实测亲和度与本地模型

`scripts/affinity_benchmark.py`从[ChEMBL官方REST接口](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)取得实测记录。本次接口状态为ChEMBL 37，2026-05-01发布。查询范围是人源PPARG（CHEMBL235/P37231）、AKT1（CHEMBL4282/P31749）、INSR（CHEMBL1981/P06213）的Ki与Kd精确nM值，非完整ChEMBL库。原始响应保存在`data/raw/public/chembl_v3`；全部请求参数、校验值、排除计数见`experiments/v3/affinity_sources.json`。

数据库由ChEMBL团队提供，采用[CC BY-SA 3.0](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions)；本项目所附来源标识、测试pK及相应派生表遵循同一许可，具体范围见`experiments/v3/DATA_LICENSE.txt`。没有提取商业软件属性计算值来拟合其计算过程。本轮使用RDKit计算的结构指纹与实测终点，而不是把关联分数当亲和度。

`scripts/local_extraction_benchmark.py`使用[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，固定版本`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，模型许可Apache-2.0。权重可从作者ModelScope镜像取得，但必须通过固定Hugging Face版本的完整SHA-256校验。推理完全在本机完成，不向远程推理服务发送病例；输入、生成输出与中间缓存保存在`outputs/research_v3`，不随Git分发。

TCMEval症状参考使用其专家Clinical Information，不使用专家证候或推理文本作为模型输入；该字段经词典投影后的24症状集合是有限参考，并不是重新人工标注的完整症状真值。失败记录、长度上限和证据引用失败计数一并报告。公开的样本哈希和混淆计数仅用于复算，不含病例原文。

## 第四轮的分区、约束与来源隔离

症状抽取主要指标改为直接对照作者独立专家短语，不再以24词表投影作为主要医学参考。50条验证记录用于选择输出方案，50条测试记录统一评价全部候选；专家答案不进入生成提示词。作者训练/验证/测试输入的规范化文本精确重叠为0，但无法排除改写重叠或大模型预训练暴露。人工存在性标注不足，未报告伪造的存在性准确率。

规划研究使用公开图谱的症状检索组合，按症状组分640/128/256条请求；两个配置分别为常规检索和带人工请求约束的检索。这些请求不是构造患者或临床疗效标签，指定配对限制也不是药理禁忌证据。精确监督只针对训练请求的证据目标，不用于训练测试请求。模型权重见`models/planning_v4`，对应来源与图谱指纹记录在训练结果中。

亲和度研究复用第三轮468条ChEMBL实测快照，增加来源文献及“文献＋骨架”的连通分组。固定三分区不足的结果完整保留；另立方案逐组留出，固定上一轮的Ridge参数，报告覆盖率和所有跳过折。由于参数曾在重叠数据上选择，逐组留出属于探索性检验。来源、许可与再分发要求见`experiments/v4/DATA_LICENSE.txt`。

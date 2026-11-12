# Data Layout

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

正式分析现在使用 `build_evidence_bundle.py` 取得的SymMap关系与STRING关联。语义指纹排除下载时间，关系与显著性字段保持可追溯。TCMSP用于独立路径核查；亲和度预测、完整实体对齐与原论文同病例评价仍未完成。本次导入不包含原论文的108,746条门诊记录或50例专家标注，也不构成论文实验复现。

## 第二轮公开病例与通路研究

新研究独立使用以下来源，未混入生产图谱，病例原文与训练文本模型不随Git分发：

| 来源 | 实际取得与用途 | 许可与引用 |
| --- | --- | --- |
| TCMEval-SDT | 作者仓库的Train_TCM_Data_v1.json，200条带答案记录；冻结图谱的外部零样本审计及探索性迁移 | 作者说明数据CC BY 4.0；Wang等，Scientific Data 12, 437 (2025)，https://doi.org/10.1038/s41597-025-04772-9；数据DOI https://doi.org/10.6084/m9.figshare.27184596.v4 |
| TCM-SD | 作者仓库固定提交72a89518392ec28f5dc1c8ed252a6c127ca2145b；train/dev/test为43,180/5,486/5,486条；监督文本分类对照 | 数据CC BY-NC-SA 4.0，作者README将其与MIT代码许可区分；Ren等，CCL 2022，https://aclanthology.org/2022.ccl-1.80/；https://github.com/Borororo/ZY-BERT |
| STRING 12.0 enrichment | 3个固定查询×3种策略×2种背景；缓存实际API响应和指纹，报告实际返回的KEGG/Reactome条目 | STRING官方 https://string-db.org/help/api/；请求记录见experiments/v2/pathways.json |

TCMEval-SDT作者测试/验证分区不公开完整答案，本项目没有伪造其答案；使用200条公开有标注分区时明确其身份，不称官方测试分区。TCM-SD只把主诉、病情描述、检查文本作为输入；标识、疾病名称和证候字段不作为特征。重复文本与源user_id重叠计数保留在报告中，后者含义未核实，尚不能保证患者级独立划分。

公开病例的辨证参考标签不是中药疗效或亲和度标签，也不能补出原论文的特定糖尿病病例。生产入口的证候模块仍使用公开图谱，研究分类器不自动替换临床工作流。

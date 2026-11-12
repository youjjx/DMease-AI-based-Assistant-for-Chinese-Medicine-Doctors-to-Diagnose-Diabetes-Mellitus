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

`knowledge_graph.json`、`examples/patients.jsonl`、`examples/triples.jsonl` 和 `examples/herb_target_affinity.csv` 是人工构造的非临床数据。亲和度数值不是实验测量或由公开数据库训练出的预测值。KAN/PPO 的已发布权重由这些构造关系生成的任务训练而来。

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

公开数据查询与非临床推荐数据目前分开。逐步接入推荐前仍需跨库标识核验、症状本体对应、关系证据分级和独立评价。本次导入不包含原论文的108,746条门诊记录或50例专家标注，也不构成论文实验复现。

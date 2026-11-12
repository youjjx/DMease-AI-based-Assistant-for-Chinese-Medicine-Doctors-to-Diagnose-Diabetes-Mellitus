# DMease中医糖尿病智能辅助诊疗系统 V1.0

DMease 是围绕同名论文持续完善的研究辅助系统。当前正式分析使用从官方入口取得的 SymMap / STRING 公开关系，TCMSP 提供可追溯的成分—靶点核查。数据不足时保留缺口，不生成替代知识或亲和度数值。

## 安装与启动

使用 Python 3.10+，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/import_public_data.py
python scripts/build_evidence_bundle.py
python -m streamlit run app.py
```

前两个数据步骤首次需要联网。原始文件与索引存放在忽略上传的 `data/raw/` 和 `data/processed/`；普通 Git 克隆不包含这些文件，须完成初始化。分析时从本地证据读取，不把患者输入发送到数据库网站。默认仅监听 `127.0.0.1:8501`。

首次运行创建操作账号。密码为12—128位并包含字母与数字；仓库没有默认账号或密码。部署提供一个初始化操作账号，没有开放注册或多角色管理。

## 已实现的操作流程

| 页面 | 功能 |
| --- | --- |
| 首页 | 知识规模、实际策略、当前账号记录、工作入口 |
| 辅助分析 | 输入资料 → 阳性/否定症状及公开证候关联 → 逐步候选与来源 → 约束核查、保存与导出 |
| 患者记录 | 查询、复用、版本关联、对比、确认删除和导出 |
| 知识图谱 | 症状—证候关系、共享症状路径、靶点关系、官方公开目录查询 |
| 研究复现 | 与辅助分析共用同一环境和策略；对照KAN/PPO与证据增益，显示梯度、轨迹和离线评价 |
| 系统与账号 | 模型状态、知识版本、密码修改和使用范围 |

支持JSON、CSV、HTML打印报告。保存前需核查；重复保存同一次分析不会重复插入。复用后产生新记录，原记录保留。输入草稿保留在当前会话；退出、刷新或关闭会话后草稿不再保证保留。持久记录保存在 `runtime/patients.db`。

自动约束覆盖填写的过敏名称、明确禁用、输入的配伍排除、妊娠状态。妊娠情况下停止生成候选，因为现有数据不提供适用性依据。公开关联库并非完整禁忌、相互作用或处方数据库，未命中约束不等于安全。年龄、体质、病史和当前用药保留为输入记录，尚无足够证据用于学习个体化药效。

## 正式推荐使用的证据

本次构建范围包含24个症状、19味名称匹配的中药、111个关联证候、48条症状—药材关系、203条症状—证候关系。取得889条症状—靶点、11,441条药材—靶点关系，并保留全部原始显著性与推断类型。满足FDR(BH)<0.05且症状与药材共同覆盖的靶点为49个；对应STRING范围内有52条高置信度关联。

统计是**当前查询范围**，不是整个数据库或原论文最终图谱规模。生地黄没有被自动合并为地黄；麦冬在SymMap中有可核查条目，已进入该来源的计算范围。TCMSP未匹配项仍保留其来源特有的缺口。

- `data/public_registry.json`：官方文件下载、字节数、指纹及索引规模。
- `data/evidence_build_report.json`：实际关系请求、范围、未匹配项与语义指纹。
- `data/processed/evidence_bundle.json`：正式计算数据，独立存放，不随Git分发。
- `models/public_model_manifest.json`：实际模型指纹、训练参数与评价位置。

原始构造图谱仅作为历史回归对照保留。其症状/证候权重、机制关系、亲和度数值和模型均不进入正式推荐。构建程序只使用其中的词表定义查询范围和声明的文字别名，随后从官方响应获得实际关系；该文字对应仍有待专业核查。

## 统一的逐步计算与解释

状态包含原始症状查询、未覆盖症状证据、相关靶点及覆盖、已选药材和硬约束可用性。每选一味都更新状态、重新计算候选，达到数量上限、没有有效证据增益或全部被约束排除时停止。

优化目标是公开关联的**证据覆盖**：症状关联覆盖权重0.65、相关靶点覆盖0.30、PPI关联覆盖0.05；缺少相应证据时重新归一化，每味候选成本0.02。这些是公开的工程参数，均不是临床效应参数。关联按存在性建模，不把FDR、SVM、RF或STRING综合分数当成结合概率，不把覆盖变化称为症状缓解。

提供两种同源策略：

1. 公开证据增益：每步选择当前边际目标增益最大的可用候选。
2. KAN/PPO：边函数网络输出逐步策略，训练与推理使用同一环境和动作掩码；可输出对当前未覆盖症状状态的局部梯度。

默认策略依据**验证集**与证据增益基线的比较选择。当前验证结果未支持KAN/PPO优于基线，因此默认使用证据增益；可在输入页或研究页明确选择KAN/PPO对照。缺少模型时仍仅使用真实公开证据；缺少或损坏证据包时停止初始化，绝不退回构造图谱。梯度是模型敏感度，不是医学因果解释。

## 训练与评价

```bash
python train.py --iterations 120 --warmup 60 --seed 42
pytest
```

已发布模型使用464个训练查询、49个验证查询、62个测试查询。查询来自本次公开症状范围的单项、双项与确定性抽样的三项组合；它们是检索任务，**不是患者病例**。知识关系不由程序生成，训练任务只选择要查询哪些已存在的实体。

按完整查询组合的稳定指纹先划分，再训练。先用训练集上的证据增益策略预训练60轮，随后执行120轮PPO，在验证集选择PPO后的检查点。另以随机种子43、44重复训练，结果分别保存在 `models/evaluation_seed43.json` 和 `models/evaluation_seed44.json`。主版本固定使用种子42，不按测试结果挑选种子。

主版本在62个测试查询上的结果：

| 策略 | 症状证据覆盖 | 综合目标 | 平均候选数 |
| --- | --- | --- | --- |
| KAN/PPO | 0.8683 | 0.8183 | 3.7903 |
| 证据增益 | 0.8683 | 0.8212 | 3.6290 |
| 随机可用候选 | 0.8548 | 0.7897 | 4.7419 |

KAN/PPO没有超过证据增益基线，故不宣称优势。约束压力测试未出现明确禁用项入选；该结果只覆盖测试的约束，不代表完整临床安全性。底层图谱在训练/验证/测试间共享，指标也由同一图谱计算，因此这是查询泛化与工程验证，不能解释为未见知识泛化、临床准确率或治疗效果。

## 扩展实验（2026-11-12）

系统“研究复现 → 实验评估”可直接查看结果与下载完整报告，无需先运行单次查询。可阅读 [独立HTML实验报告](experiments/report.html)（下载后用浏览器打开）、[汇总表](experiments/ablation_summary.csv)和[原始结果](experiments/results/summary.json)。

已完成7种方法×3随机种子、14组参数设置、18组证据删除试验、1,488次约束查询执行，以及62个测试查询的1,247,121个可行子集枚举。全部方法在同一个完整参考图谱与默认目标下评分。

| 方法 | 综合目标（三种子均值） | 种子间样本标准差 |
| --- | --- | --- |
| KAN + PPO | 0.818226 | 0.000142 |
| MLP + PPO | 0.819213 | 0.000642 |
| 移除PPI | 0.817284 | 0.001613 |
| 移除直接关系 | 0.641137 | 0.010754 |
| 移除靶点及PPI | 0.551825 | 0.025414 |
| 仅模仿预训练 | 0.820263 | 0.000559 |
| 仅PPO | 0.812250 | 0.003034 |

确定性证据增益基线为0.821231。当前结果不支持KAN/PPO优于MLP、仅模仿预训练或贪心基线；PPI增量也未得到充分支持。精确枚举发现7/62个查询存在更优集合，平均目标提升0.002238。1,488次执行未违反指定排除规则，但这些是同一查询的多场景重复检查，不是临床安全性样本。

靶点与PPI证据分别在53/62、41/62个查询中可用；基线在有证据查询上的覆盖分别为0.9916、1.0000。报告分开列出证据可用比例与条件覆盖，避免把缺失数据混作算法表现。配对区间和校正p值也已提供，但查询组合有重叠，统计独立性未建立，只作条件性描述。

症状解析修复了否定范围、当前状态和词语包含造成的重复识别。164条开发语言用例完全一致数从80变为164；用例参与了规则修复，**不是独立临床测试集**。复杂主体、假设、不确定性和时间表达仍需进一步评价。

```bash
python scripts/run_experiments.py --seed 42
python scripts/run_experiments.py --seed 43
python scripts/run_experiments.py --seed 44
python scripts/run_experiments.py --aggregate
python scripts/evaluate_parser.py --label after
python scripts/verify_experiment_artifacts.py
pip install -e ".[research]"
python scripts/build_experiment_report.py
```

实验方案在本轮运行前固定于 `experiments/protocol.json`，不是外部预注册。结果记录代码、数据和模型指纹；训练需要按前文完成数据初始化，并与方案的证据指纹一致。报告生成和结果核验不需下载原始数据库。运行结果来自Python 3.13.7、PyTorch 2.7.1、NumPy 2.4.1、CPU单线程；换环境可能产生差异，不能覆盖原报告后继续当成同一次实验。试验权重保存在忽略的 `outputs/experiments/` 中，可按固定方案重新训练；生产模型未按这次测试结果替换。

代码指纹保留实际运行时的字节校验值；离线核验仅容许Git跨平台检出产生的LF/CRLF换行等价，其他内容改变仍会失败。本轮测试查询已被分析，后续方法迭代需要新的冻结评价与独立材料。完整的实施方法、局限及逐查询误差对照见报告。

## 第二轮改进与公开病例评价

“研究复现 → 改进与外部评价”提供策略对照、真实病例辨证指标、通路与靶点一致性，以及可交互的单次集合优化。[完整第二轮报告](experiments/v2/report.html)、[数值汇总](experiments/v2/summary.json)和[监督辨证结果](experiments/v2/supervised_syndrome.json)均可核查。

新增三次B样条KAN、GAE与价值预热、独立学习率、KL提前停止、衰减熵、允许保留预训练检查点，以及仅训练查询上的精确集合示范。保持原来的参考目标；全部12个新模型先按新验证集选定，再对与上一轮不重叠的192个查询统一测试。

| 原始策略 | 新测试综合目标（三种子均值） |
| --- | --- |
| 旧版RBF/KAN-PPO | 0.821893 |
| 稳定RBF方案 | 0.824199 |
| 三次样条KAN方案 | 0.823329 |
| 最优集合示范与样条 | 0.822635 |
| 最优集合示范与MLP | 0.822870 |
| 确定性证据增益 | 0.825261 |

稳定方案将与贪心的差距缩小约68%，但未超过贪心。其三个种子均选择PPO前检查点，不能将收益归因于PPO。相同的删除/增补/交换搜索提供给所有方法；贪心搜索后为0.826407，精确上界0.826615，190/192个查询达到集合最优。生产模型未按新测试结果替换。

补算论文形式的TCS/Jaccard后，贪心在共享49靶点范围为0.2158，在全部取得的显著关联靶点范围仅0.0061。二值关联覆盖不是预测亲和度；大量非查询靶点会扩大并集。STRING 12.0通路富集也已执行：以当前可选靶点为背景未返回显著通路，整个人类背景下则有较多条目；同一查询的三种策略产生相同靶点集合，不能证明PPI的独立增益。

实际取得了两套作者公开的标注数据，未将其病例原文放入Git：

- **TCMEval-SDT**：200条带答案的公开训练分区，用于冻结系统的外部零样本审计。原图谱前三候选命中2/200；335个金标准证候标签中只有28个能按固定文字规则映射，揭示适用范围和标签对齐缺口。
- **TCM-SD**：43,180条训练、5,486条验证、5,486条测试，148类证候。只用主诉、病情描述和检查字段，按去除训练重复文本后的验证macro-F1选择监督文本分类器。官方测试准确率**82.46%**、macro-F1 **0.4247**、Kappa **0.8085**；排除与训练/验证重复的测试文本后，4,888条上的准确率为**80.91%**、macro-F1 **0.4113**。不均衡类别和跨数据集迁移仍明显受限，不能只报告准确率。

这是监督文本对照，不是LLM或KAN/PPO。它迁移至TCMEval-SDT时首候选命中4%、前三命中7.5%，说明同域成绩不能代表跨域临床适用性。两套数据均不等同于原论文50例糖尿病病例；82.46%不能冒称复现论文86%。测试中有源user_id跨分区重复，其含义未核实，不宣称患者级独立评价。

复现先完成公开证据初始化，再运行以下命令。报告及数值核验不需病例原文；训练和外部评价会取得公开数据，原始文件和研究模型保存在忽略目录中。建议在独立检出目录重跑，以保留发布结果。

```bash
python scripts/research_v2.py --prepare
# 首次检出时重建旧策略对照权重；已有权重可省略这三行
python scripts/reproduce_policy_v2.py --legacy --seed 42
python scripts/reproduce_policy_v2.py --legacy --seed 43
python scripts/reproduce_policy_v2.py --legacy --seed 44
python scripts/reproduce_policy_v2.py --seed 42
python scripts/reproduce_policy_v2.py --seed 43
python scripts/reproduce_policy_v2.py --seed 44
python scripts/research_v2.py --aggregate
python scripts/pathway_audit.py
python scripts/external_syndrome_audit.py
python scripts/tcm_sd_benchmark.py
python scripts/verify_v2.py
python scripts/report_v2.py
```

第二轮尚未覆盖实测亲和度与实际语言模型调用；第三轮继续开展这两项研究。体质疗效适配和原病例临床评价仍缺少对应材料。各轮方法差距、数据缺口和未支持论文结论的结果均保留。

## 第三轮：实测亲和度、语言模型抽取与迁移诊断

“研究复现 → 亲和度与模型抽取”展示本轮四个研究项目。[完整报告](experiments/v3/report.html)、[固定方案](experiments/v3/protocol.json)、[实测亲和度结果](experiments/v3/affinity.json)、[抽取结果](experiments/v3/extraction.json)和[迁移诊断](experiments/v3/syndrome_diagnostics.json)包含可复算的记录数与误差统计。

- **ChEMBL 37实测Ki/Kd**：从官方接口取得PPARG、AKT1、INSR三靶点数据，筛选高置信人源单蛋白实验；标准化结构、聚合重复测量，以Morgan指纹预测pK。按分子骨架固定划分，不混合Ki与Kd。只有PPARG-Ki满足预设样本数，训练/验证/测试为181/40/32个化合物。验证选出的岭回归测试MAE为0.434 pKi，训练均值基线为0.801；R²为0.426。其他五个终点全部保留，未为达到样本量更改划分。
- **真实本地语言模型调用**：固定Qwen2.5-1.5B-Instruct权重版本，对TCMEval-SDT的200条公开记录单次推理；不输入专家答案，核验JSON、原文引用与24症状词表。91条JSON失败；只去除完整对象后的多余引号后，仍有33条失败，字段缺失和语义映射问题仍存在。该模型/提示词组合目前不能替换规则解析。参考由同一规则词典投影，会偏向规则方法，不能作为公平的医学抽取排名或临床金标准。失败输出计入评价，原文和生成文本留在忽略目录。
- **冻结分类器诊断**：外部200条记录中只有52条包含分类器已知证候，固定标签表的首候选命中率上限为26%；实际为4%。字符特征覆盖从同源测试的93.4%下降到41.9%。少于20条训练样本的证候测试准确率15.7%，超过或等于1000条的类别为93.6%。这些是描述性误差分析，没有基于测试重选模型。
- **少样本加权改进**：固定6个权重/正则候选与原基线，按验证宏平均F1选择平方根逆频率加权方案。完整测试准确率由82.46%变为83.41%，148类宏平均F1由0.425变为0.581。该实验复用了已看过的公开测试集，属于探索性改进，不算新的独立临床验证，也不归为KAN/PPO收益。

亲和度首次执行遇到RDKit环信息兼容性问题，误排除8条有效活动记录；修复前的代码和数值保存在[实现审计记录](experiments/v3/initial_affinity_implementation/affinity.json)，[修复说明](experiments/v3/affinity_implementation_amendment.json)列明影响。修复沿用原划分与选模规则；更正后的评价不被当作新的独立验证。

复现需先完成公开证据初始化及第二轮公开病例准备。在独立检出目录操作，保留原发布结果。额外依赖与流程：

```bash
pip install -e ".[research,affinity,local-llm]"
python scripts/reproduce_affinity_v3.py
python scripts/download_local_extractor.py --mirror modelscope
python scripts/local_extraction_benchmark.py
python scripts/extraction_format_audit.py
python scripts/syndrome_diagnostics_v3.py
python scripts/imbalance_benchmark_v3.py
python scripts/verify_v3.py
python scripts/report_v3.py
```

第三轮脚本使用Python 3.11或以上，完整版本记录见`experiments/v3/environment.json`。模型下载可省略`--mirror modelscope`使用Hugging Face；无论下载源如何，完整权重的SHA-256必须与固定官方版本一致。本轮在4GB显存的RTX 3050上按单条、半精度推理运行，所用配置和设备信息保存在报告。亲和度训练和病例分类模型仅从本机生成且校验过的文件恢复，不接收任意外部序列化模型。

亲和度复现默认使用仓库附带的468条标准化化合物—靶点—终点记录，含实测pK、结构及原始活动标识，遵循ChEMBL原许可。`affinity_benchmark.py`为原始下载和执行入口；`reproduce_affinity_v3.py`使用固定快照，避免接口更新改变实验数据，并在首次训练时归档发布的选模文件。它会比较新运行和发布结果的选模及MAE差异。

本轮化合物亲和度尚未可靠连接到中药成分；pK不是结合概率。语言模型抽取是独立研究入口，不自动替换正式辅助分析；也不代表已经复现原论文的大模型和50例病例结果。ChEMBL派生数据的署名和相同方式共享许可见[数据许可](experiments/v3/DATA_LICENSE.txt)。

## 第四轮：结构化抽取与约束规划

报告：[第四轮研究报告](experiments/v4/report.html)。完整结果、冻结方案和逐查询统计保存在`experiments/v4`。

同一50条作者测试病例上，旧四字段配置的严格专家短语F1为0.0102；验证预先选中的字段/字符表约束方案为0.4375，JSON失败3/57段。原文子串约束方案为0.5173，JSON失败0/57段；后者测试分数更高，但没有用它改写验证选模记录。词表匹配与整句切分对照分别为0.2200和0.3402。

同一256条工程检索请求上，贪心目标0.75913，精确最优0.76175。无加性先验KAN的验证选中快照均值为0.75515，保守阈值方案为0.75896，仍未优于贪心；额外将原始网络与贪心完整结果验算择优为0.76053，贪心加集合优化为0.76123。网络学习与额外搜索贡献分别报告。14,848条候选结果和256个精确解已从实际证据图重算核对，误差为0。

- **抽取故障修复与独立参考**：将语言模型的原文抽取和词表规范化分开，新增JSON结构约束、原文子串约束及保留偏移的分段。按作者原有50条验证/50条测试分区评价，主要指标直接对照专家短语；所有失败输出仍计入分母。
- **规划实现改进**：状态增加完整配对限制与剩余数量预算，新增显式停止动作，允许暂时不增加目标的可行候选。训练请求用精确后续目标监督；分别训练增益先验KAN、相近参数量MLP和无加性先验KAN，每种3个种子，保留预训练/PPO对照及全部验证快照。
- **更强且相同的对照**：在新的256条工程检索请求上统一比较原始网络、单步贪心、双步搜索、共享集合优化与精确最优。额外贪心兜底和集合搜索单独列出，不将其收益归因于KAN。
- **补充实验**：冻结辨证模型的作者测试分区评价；实测亲和度的文献隔离、文献与骨架联合隔离、固定参数逐来源组留出。逐组留出的参数曾在重叠数据上选择，属于探索性检查。

更正：之前关于“TCMEval作者验证/测试答案未公开”的判断不正确。答案实际单独保存在作者仓库`Results`目录；第四轮已取得并逐ID核验，旧轮数值与200条训练分区身份保持不变。出处、哈希和更正记录见`extraction_sources.json`。

研究界面新增“结构化抽取与约束规划”工作区：查看实验结果、核对原文抽取、指定排除/配对/数量限制并比较规划候选。规划模型9个重复训练版本的选中和预训练快照随仓库提供；语言模型使用固定版本本地权重，不向远程服务提交输入。交互核对固定使用原文子串约束以便审查引用，实验中的验证选模结果单独展示。原文引用有效不等于医学解释正确，研究结果不自动写回患者诊断。

第四轮研究脚本使用Python 3.11或以上，精确复现应使用`experiments/v4/environment.json`记录的依赖版本；改变运行库时应另建实验版本，不混用旧缓存。先按前述流程构建公开证据包、下载并校验模型及TCM-SD研究数据，再运行：

```bash
pip install -e ".[research,affinity,local-llm]"
python scripts/extraction_v4.py --prepare
python scripts/extraction_v4.py --phase validation
python scripts/extraction_v4.py --phase test
python scripts/extraction_statistics_v4.py
python scripts/planning_v4.py --prepare
python scripts/planning_v4.py --train
python scripts/export_planning_v4.py
python scripts/gating_v4.py --select
python scripts/planning_v4.py --aggregate
python scripts/gating_v4.py --test
python scripts/verify_v4_local.py
python scripts/planning_diagnostics_v4.py
python scripts/diagnosis_v4.py
python scripts/affinity_v4.py
python scripts/affinity_loco_v4.py
python scripts/report_v4.py
python scripts/verify_v4.py
```

重新训练请使用独立检出目录，保留发布结果；验证脚本可直接核查发布的汇总、模型及代码指纹。训练阶段权重在忽略目录`outputs/research_v4/planning`，发布副本位于`models/planning_v4`。离线核查不需要病例原文或第三方大模型。临床原始病例、体质效应、药材成分结构连接等差距仍见下表；未把公开检索目标视为论文疗效或诊断成绩。

## 命令行

```bash
dmease analyze --symptoms "口渴，乏力，尿频" --exclude 黄芪
```

历史合成样例仅在显式指定时运行：

```bash
dmease --legacy-nonclinical infer --patient-id P2025-0001
python scripts/train_legacy_policy.py --iterations 50 --episodes 8
```

`DMEASE_EVIDENCE_BUNDLE`可指定界面证据包；`DMEASE_CHECKPOINT`可指定模型；`DMEASE_DATA_DIR`指定账号和患者记录目录。模型须匹配实际语义指纹、实体顺序及状态结构；下载时间不影响语义一致性。

## 与论文仍存在的差距

| 论文部分 | 当前状态 |
| --- | --- |
| 知识图谱和患者界面 | 公开数据取得、逐条来源、持久记录和操作流程已接通；范围小于论文 |
| 症状和证候 | 词表、否定处理、公开关系已接通；新增本地模型结构化抽取和作者测试分区专家参考，跨数据集标签覆盖仍不足 |
| 实测亲和度 | 三靶点Ki/Kd处理与骨架隔离建模已实现；PPARG-Ki完成评价，其他终点样本不足，中药成分结构连接尚缺 |
| KAN/PPO序贯决策 | 新增完整约束状态、显式停止、精确训练监督与三种子对照；研究页可试算新模型，目标仍为证据覆盖 |
| 可解释性 | 公开路径、每步新增覆盖、状态和策略梯度已实现 |
| 体质适配 | 资料记录已实现；缺少经验证数据，不施加编造的体质药效权重 |
| 临床与论文实验 | 未取得原论文门诊记录和专家标注；未复现论文的86%准确率、0.83 F1或0.67 TCS |

完整方法复现仍需中药成分到结构与亲和度的可靠连接、独立抽取标注、体质适配证据、完整禁忌规则和原研究或等效独立病例。不能用检索指标替代这些验证。

## 账号与数据维护

密码使用随机盐与PBKDF2-HMAC-SHA256保存；连续5次失败锁定60秒，30分钟无操作后下次交互重新登录。密码修改使旧会话在下次交互时失效。数据库记录按账号访问，退出清除会话数据。部署者应在停止写入后配套备份账号和患者数据库。部署环境的HTTPS、网络访问控制由部署者配置。

## 引用与数据许可

数据获取及许可证说明见 `data/README.md`，原始数据库内容不随源码重新发布。请分别引用SymMap、STRING和TCMSP的官方文献，并注明使用版本及筛选方式。

```bibtex
@inproceedings{jia2025dmease,
  title = {DMease: AI-based Assistant Tool for Traditional Chinese Medicine Doctors in Diagnosing and Treating Diabetes Mellitus},
  author = {Jia, Junxiang and Hao, Xinyi and Hu, Yizhong and Meng, Haoyang and Zhao, Cong and Guan, Jianfeng and Sun, Weiwei},
  booktitle = {IEEE BIBM},
  year = {2025}
}
```

# 2026-09-08 评测快照

本页保留 2026-09-08 冻结快照；当前展示入口见 [关键测评](../README.md)。工具预算和评分进度按当时配置解读。

> 新增 [800 查询的混合检索与最终回答测评](../hybrid-grounding-2026-09-08/README.md)：1,016 篇语料、1,440 份 RAG 回答、288 份本地 Mem0 回答；提供开发集选参、证据不足检查、配对分析和离线复算。

新增 [训练动机与错题拆解](../training-motivation-2026-09-08/README.md)：从312题、427个前提的原始回答和评分复算错题类别，区分知识持续答错、问法不稳和应用失败；附[CPT关键词选材核查与改进方案](../training-motivation-2026-09-08/cpt_data_construction.md)。该补充包单独提供文件哈希，不改写下面的历史快照。

新增 [CPT+SFT 三项通用基准实测](../general-benchmarks-2026-09-08/README.md)：各 500 题，MMLU **61.2% → 59.0%**、ARC-Challenge **44.2% → 51.2%**、HellaSwag **65.8% → 67.4%**；提供配对区间、完整数据与离线复算。

建议先读 [Agent 关键设计与测评方法](agent_improvements.md)，了解每项设计解决的问题、对照数据和测量方式；再读 [整体验收](agent_holdout.md) 与 [架构对照](architecture.md)。

新增 [RAG / Mem0 独立组件测评](../component-benchmark-2026-09-08/README.md)：公开 1,016 篇语料、400 条 RAG 查询、120 个 Mem0 场景、真实 API 记录和完整复算脚本。[RAG](../component-benchmark-2026-09-08/rag/README.md)给出 Top-K、阈值与上下文对照，[Mem0](../component-benchmark-2026-09-08/memory/README.md)给出真实抽取、重开后检索和逐事实评分。新套件单独冻结与校验，下方原快照继续保留。

另公开 [72 条 RAG 开发回归探针及离线复算](../rag-2026-09-08/README.md)：包含最初 60 题与 12 条近义无答案题、15 个资料块、实际查询向量和完整排名，用于查看上下文补全、类型过滤及阈值对检索的影响。它与上述 400 条查询套件、72 个 Agent 整体场景分别统计。

这里保存精选结果与可追溯状态。41 份结构化文件可直接浏览，434 份详细执行记录见 [下载结果附件](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-08/evaluation-details-2026-09-08.zip)（约 13.03 MiB）。各实验按独立快照归档，历史附件提供追溯入口。

| 内容 | 报告 | 机器可读证据 |
| --- | --- | --- |
| Agent 关键设计：并行、检索、记忆、复用、预算 | [设计与测量方法](agent_improvements.md) | 报告逐项链接汇总、组件测试及原始轨迹；历史补查例见 `agent/improvement_v2/runs/N08__*` |
| 72 场景 × 3 模型及固定重复 | [整体验收方法与结果](agent_holdout.md) | [分层分母](agent/holdout_v1_live_20260908/holdout_metrics.json)、[运行遥测](agent/holdout_v1_live_20260908/telemetry_summary.json)；`runs/` 运行结果与 `errors/` 执行诊断 |
| 单/多 Agent、Worker 串并行 | [架构对照](architecture.md) | [汇总](agent/architecture_compare_20260908_v2/comparison_summary.json)、`runs/`、`scheduler/` |
| CPT/SFT/RL 与知识调用 | [机制报告](model_diagnostics.md) | [312 题配对](training/mechanism_diagnosis_20260908/mechanisms/analysis.json)、[逐题状态](training/mechanism_diagnosis_20260908/mechanisms/states.json) |
| 冻结 128 个 RL 训练题 | [训练题完整统计](training/mechanism_diagnosis_20260908/rl_frozen_eval/analysis.json) | [逐题状态](training/mechanism_diagnosis_20260908/rl_frozen_eval/states.json)、[五道改善复核](training/mechanism_diagnosis_20260908/rl_frozen_eval/train_greedy_gain_review.json) |
| 600 题旧模型选择题诊断 | [原标签结果](training/medical_holdout600_v1/results.json) | [输入审查](training/medical_holdout600_v1/input_quality_review.json)、[语义评分进度](training/medical_holdout600_v1/semantic_scoring_status.json) |

机器结果中的 `runs/...`、`scheduler/...`、`errors/...` 等路径位于 ZIP 对应实验目录；本目录保留汇总与模型逐题状态，没有打包全部服务请求和历史评分尝试。模型 `states.json` 提供逐题评分与干预状态，原始生成全文只在部分人工式复核材料中保留。费用包含模型和检索的实验观测值，不是当前供应商报价。

## 计数与验证

[manifest.json](manifest.json) 记录精选源文件与公开副本各自 SHA-256；路径已规范化，账号标识匿名化，凭据和私有运行字段已排除。文件清单与异步评分进度分别记录。600 题评测对应既有权重，新增权重链按其独立实验结果解读。

下载 ZIP 后，从仓库根目录运行 `python results/verify_september08.py --assets-dir path/to/downloads`，校验本次文件并复核主要汇总和计数关系。该脚本不调用模型、不重新评判医学正确性。阅读首页应同时检查分母、评分状态、训练/开发/诊断用途与单种子限制。

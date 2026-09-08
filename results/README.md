# 完整评测结果

新增 [CPT+SFT 科学推理基准实测](general-benchmarks-2026-09-08/README.md)：ARC-Challenge 固定 500 题，`acc_norm` **44.2% → 51.2%（+7.0 个百分点）**；提供三项基准完整数据与离线复算。

**最新增量：[2026-09-08 新评测与代码更新](2026-09-08/README.md)**。以下为继续保留的 9 月 7 日历史发布。

2026-09-07 结果快照。包含 35 组结果目录、14,055 份结构化文件：正式留出评测、开发实验、消融、历史诊断与性能探针。目录数和文件数不是独立题目数。

[下载全部结果附件](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/tag/evaluation-results-2026-09-07) · [文件与版本清单](index.json) · [Agent 逐任务结果](agent_cases.csv) · [模型分组指标](model_scores.csv)

## 主要结果

| 评测 | 结果 | 可浏览明细 |
| --- | --- | --- |
| Agent 问诊 | 48 个任务、96/96 条有效双评审；两位评审均全项通过：MiniMax 13/24，Qwen 14/24 | [完整评分与理由](summaries/agent/full_system_v1/campaign_summary.json) |
| LoRA 留出评测 | 247 题、944 份回答；知识与上下文，单训练种子 | [分组与配对区间](summaries/training/independent_holdout_20260906/results.json) |
| T0–T3 | 150 道知识开发题、16 张 VQA 开发图像；答案形式与关系解释对照 | [完整指标](summaries/training/format_factorial_v1/results.json) |
| GSPO | 93 题、16 张验证图像；归一化评分 60.75 → 66.13，9 改善 / 2 退步 / 82 同分 | [逐题配对](summaries/training/gspo_vqa_rl_v1/comparison.json) |
| CPT + SFT | CPT + SFT 与仅 SFT 的分任务对照；排除题目与配对区间见文件 | [分组指标与局限](summaries/training/cpt_medical_v1/results.json) |
| RAG 边界 | 主套件 60 个场景；另有 12 个近似但不可回答的问题 | [主套件](summaries/agent/full_system_v1/rag_boundary_v1/summary.json)、[近似问题](summaries/agent/full_system_v1/rag_near_miss_v1/summary.json) |
| Mem0 边界 | 相同 12 个场景重复 3 轮，采用词汇锚点判定 | [重复实验汇总](summaries/agent/full_system_v1/mem0_repeat_summary/summary.json) |

分数来自自动评审或协议检查，不能合并为临床准确率。Mem0 重复实验不是 36 个独立病例；RAG 返回相关块也不等于回答正确。问诊主评测、独立的 Mem0 测试和记忆消融采用不同协议，不能互相替代。历史生成、重试、修订评分均按原目录保留；主问诊结果以 `campaign_summary.json` 中列出的运行和最终评审为准。

## 下载内容

| 附件 | 内容 | 大小 |
| --- | --- | --- |
| [agent-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/agent-evaluations.zip) | Agent 对话、最终与历史评分、执行轨迹、RAG/Mem0 边界及记忆消融 | 2.92 MiB |
| [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) | 模型逐题生成与评分、CPT/SFT/GSPO、配方对照、校准及历史实验 | 23.78 MiB |
| [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) | 诊断与性能探针，包括失败和未完成运行的状态 | 4.69 MiB |

附件保留实验内的相对目录，每组附 `PUBLICATION_MANIFEST.json`，记录源文件和公开文件各自的 SHA-256。机器路径已转换，测试身份已一致化匿名处理，凭据及账户状态字段不包含在结果中；模型权重、完整 CPT 训练语料、预处理资料池、提示词临时文件和内部会议记录不属于本结果集。原始文件中的状态与错误保留，不将未完成运行补记为成功。

## 校验与复算

下载三个 ZIP 后，从仓库根目录执行（Python 3.11+，仅标准库）：

```powershell
python results/verify.py --assets-dir path/to/downloads
```

脚本校验全部附件和 14,055 个文件的哈希，并从逐题记录复算 Agent、LoRA 留出、T0–T3 与 GSPO 的主要计数/分数。它不重跑模型，也不重新确认医生级正确性；其他历史实验保留原汇总与明细，不声称其全部统计均已独立复算。完整运行轨迹包含的是模拟患者和测试状态。

`cpt_diagnosis_v3` 在抓取后仍有状态更新，本次只发布抓取时快照。部分目录没有最终汇总，或仅提供计划/中断状态，表示该运行尚无完整结论。各实验的模型、划分、来源版本及生成参数以其配置和协议文件为准；历史源码哈希可能不同于当前代码版本。

## 全部目录

| 目录 | 类型 | 文件数 | 可直接浏览的汇总 | 附件 |
| --- | --- | ---: | --- | --- |
| `full_system_v1` | 工程与问诊开发评测 | 1195 | [campaign_summary.json](summaries/agent/full_system_v1/campaign_summary.json)、[component_contracts.json](summaries/agent/full_system_v1/component_contracts.json) | [agent-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/agent-evaluations.zip) |
| `analysis` | 模型评测与实验 | 20 | 见附件内结果与状态 | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `base_calibration_bounded_v2` | 模型评测与实验 | 117 | [status.json](summaries/training/base_calibration_bounded_v2/status.json)、[verification.json](summaries/training/base_calibration_bounded_v2/verification.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `base_calibration_v1` | 模型评测与实验 | 240 | [status.json](summaries/training/base_calibration_v1/status.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `cpt_diagnosis_v1` | 诊断与性能探针 | 124 | [status.json](summaries/training/cpt_diagnosis_v1/status.json)、[plan.json](summaries/training/cpt_diagnosis_v1/plan.json) | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `cpt_diagnosis_v2` | 诊断与性能探针 | 15 | [status.json](summaries/training/cpt_diagnosis_v2/status.json)、[plan.json](summaries/training/cpt_diagnosis_v2/plan.json) | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `cpt_diagnosis_v3` | 诊断与性能探针 | 1783 | [status.json](summaries/training/cpt_diagnosis_v3/status.json)、[plan.json](summaries/training/cpt_diagnosis_v3/plan.json) | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `cpt_medical_v1` | 模型评测与实验 | 2088 | [results.json](summaries/training/cpt_medical_v1/results.json)、[status.json](summaries/training/cpt_medical_v1/status.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `direct_evaluations` | 模型评测与实验 | 27 | [summary.json](summaries/training/direct_evaluations/all/summary.json)、[summary.json](summaries/training/direct_evaluations/all_concise/summary.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `evaluations` | 模型评测与实验 | 76 | [summary.json](summaries/training/evaluations/all/summary.json)、[summary.json](summaries/training/evaluations/all_concise/summary.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `experiments` | 模型评测与实验 | 24 | 见附件内结果与状态 | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `format_factorial_v1` | 模型评测与实验 | 1101 | [results.json](summaries/training/format_factorial_v1/results.json)、[final_verification.json](summaries/training/format_factorial_v1/final_verification.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `gspo_gpu_batch_benchmark_v1` | 诊断与性能探针 | 5 | 见附件内结果与状态 | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `gspo_infra_probe_v1` | 诊断与性能探针 | 7 | 见附件内结果与状态 | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `gspo_infra_probe_v2` | 诊断与性能探针 | 14 | [summary.json](summaries/training/gspo_infra_probe_v2/summary.json) | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `gspo_infra_probe_v3` | 诊断与性能探针 | 9 | [summary.json](summaries/training/gspo_infra_probe_v3/summary.json) | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `gspo_judge_batch16_v1` | 模型评测与实验 | 5 | [summary.json](summaries/training/gspo_judge_batch16_v1/summary.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `gspo_judge_concurrent_v1` | 模型评测与实验 | 4 | [summary.json](summaries/training/gspo_judge_concurrent_v1/summary.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `gspo_judge_models_astra_v1` | 模型评测与实验 | 5 | [summary.json](summaries/training/gspo_judge_models_astra_v1/summary.json)、[summary.json](summaries/training/gspo_judge_models_astra_v1/gpt-6-astra/summary.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `gspo_judge_models_v1` | 模型评测与实验 | 7 | [summary.json](summaries/training/gspo_judge_models_v1/summary.json)、[summary.json](summaries/training/gspo_judge_models_v1/gpt-5.3/summary.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `gspo_judge_models_v2` | 模型评测与实验 | 23 | [comparison.json](summaries/training/gspo_judge_models_v2/comparison.json)、[summary.json](summaries/training/gspo_judge_models_v2/summary.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `gspo_vqa_rl_v1` | 模型评测与实验 | 352 | [comparison.json](summaries/training/gspo_vqa_rl_v1/comparison.json)、[status.json](summaries/training/gspo_vqa_rl_v1/status.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `independent_holdout_20260906` | 模型评测与实验 | 1488 | [results.json](summaries/training/independent_holdout_20260906/results.json)、[status.json](summaries/training/independent_holdout_20260906/status.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `knowledge_atomic_v1` | 诊断与性能探针 | 114 | [results.json](summaries/training/knowledge_atomic_v1/results.json) | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `knowledge_experiments_v1` | 模型评测与实验 | 233 | [report_metrics.json](summaries/training/knowledge_experiments_v1/analysis/report_metrics.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `knowledge_experiments_v1_probe_bs2` | 诊断与性能探针 | 3 | 见附件内结果与状态 | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `knowledge_experiments_v1_probe_bs4` | 诊断与性能探针 | 3 | 见附件内结果与状态 | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `knowledge_retention_v1` | 模型评测与实验 | 2151 | [results.json](summaries/training/knowledge_retention_v1/results.json)、[verification.json](summaries/training/knowledge_retention_v1/verification.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `native_experiments` | 模型评测与实验 | 1 | 见附件内结果与状态 | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `native_verification` | 诊断与性能探针 | 22 | 见附件内结果与状态 | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `open_qa_verification` | 诊断与性能探针 | 6 | 见附件内结果与状态 | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `original_resolution_verification` | 诊断与性能探针 | 13 | 见附件内结果与状态 | [model-diagnostics-and-probes.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-diagnostics-and-probes.zip) |
| `rationale_pilot_v1` | 模型评测与实验 | 355 | [results.json](summaries/training/rationale_pilot_v1/results.json)、[verification.json](summaries/training/rationale_pilot_v1/verification.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `rationale_pilot_v2` | 模型评测与实验 | 572 | [results.json](summaries/training/rationale_pilot_v2/results.json)、[verification.json](summaries/training/rationale_pilot_v2/verification.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |
| `reasoning_effects_v1` | 模型评测与实验 | 1853 | [diagnostics_results.json](summaries/training/reasoning_effects_v1/diagnostics_results.json)、[pilot_aggregate_stats.json](summaries/training/reasoning_effects_v1/pilot_aggregate_stats.json) | [model-evaluations.zip](https://github.com/qhzeng-gittec/medical-agent-assistant/releases/download/evaluation-results-2026-09-07/model-evaluations.zip) |

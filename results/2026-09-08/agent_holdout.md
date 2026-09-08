# 独立合成留出集：执行与盲评报告

本报告是开发测评，不提供临床正确率认证。计划的主样本是 72 个合成场景；模型、轮次和重复执行均不是新增独立病例。
两个固定评审分别只看到检查点时点及以前的实际对话、状态与工具结果，看不到目标模型身份、隐藏患者事实或未来脚本。
每个时点的双评审独立调用，分歧保留为 uncertain；任何一位评审发现的关键失败另行计数，仍需独立复核。
MiniMax 和 Qwen 同时参与目标执行与评审时可能存在同源偏差。合成患者和模型评审均不能替代医生审核。

完成率为适用检查项全部通过的运行数 / 已完整评分运行数，并同时给出计划分母。未执行、服务不可用、评分错误、uncertain 与 not_applicable 单列；全不适用不算完成。
运行基础设施错误不记临床失败。控制检索/记忆仅检验受控输入的使用，不能冒充真实服务端到端质量。

## 主样本（每模型每案例 repetition=1，恢复尝试单列）

| 模型 | 领域 | 难度 | 已执行/计划 | 完成评分 | 全适用项通过/评分（计划） | 关键失败/已评运行 | 分歧项 |
|---|---|---|---:|---:|---:|---:|---:|
| gemini-3.5-flash | all | all | 72/72 | 51 | 43/51（72） | 7/67 | 11 |
| gemini-3.5-flash | all | boundary | 24/24 | 17 | 15/17（24） | 2/22 | 4 |
| gemini-3.5-flash | all | complex | 24/24 | 14 | 11/14（24） | 2/21 | 2 |
| gemini-3.5-flash | all | routine | 24/24 | 20 | 17/20（24） | 3/24 | 5 |
| gemini-3.5-flash | consultation | all | 24/24 | 17 | 13/17（24） | 3/24 | 4 |
| gemini-3.5-flash | consultation | boundary | 8/8 | 6 | 6/6（8） | 0/8 | 0 |
| gemini-3.5-flash | consultation | complex | 8/8 | 5 | 3/5（8） | 1/8 | 1 |
| gemini-3.5-flash | consultation | routine | 8/8 | 6 | 4/6（8） | 2/8 | 3 |
| gemini-3.5-flash | evidence | all | 24/24 | 18 | 15/18（24） | 3/22 | 5 |
| gemini-3.5-flash | evidence | boundary | 8/8 | 7 | 5/7（8） | 2/7 | 4 |
| gemini-3.5-flash | evidence | complex | 8/8 | 4 | 3/4（8） | 1/7 | 1 |
| gemini-3.5-flash | evidence | routine | 8/8 | 7 | 7/7（8） | 0/8 | 0 |
| gemini-3.5-flash | history | all | 24/24 | 16 | 15/16（24） | 1/21 | 2 |
| gemini-3.5-flash | history | boundary | 8/8 | 4 | 4/4（8） | 0/7 | 0 |
| gemini-3.5-flash | history | complex | 8/8 | 5 | 5/5（8） | 0/6 | 0 |
| gemini-3.5-flash | history | routine | 8/8 | 7 | 6/7（8） | 1/8 | 2 |
| minimax/minimax-m2.5 | all | all | 72/72 | 57 | 33/57（72） | 22/68 | 25 |
| minimax/minimax-m2.5 | all | boundary | 24/24 | 20 | 14/20（24） | 6/23 | 8 |
| minimax/minimax-m2.5 | all | complex | 24/24 | 18 | 6/18（24） | 10/22 | 11 |
| minimax/minimax-m2.5 | all | routine | 24/24 | 19 | 13/19（24） | 6/23 | 6 |
| minimax/minimax-m2.5 | consultation | all | 24/24 | 19 | 8/19（24） | 12/23 | 13 |
| minimax/minimax-m2.5 | consultation | boundary | 8/8 | 7 | 6/7（8） | 2/8 | 0 |
| minimax/minimax-m2.5 | consultation | complex | 8/8 | 7 | 1/7（8） | 5/8 | 9 |
| minimax/minimax-m2.5 | consultation | routine | 8/8 | 5 | 1/5（8） | 5/7 | 4 |
| minimax/minimax-m2.5 | evidence | all | 24/24 | 20 | 11/20（24） | 6/24 | 10 |
| minimax/minimax-m2.5 | evidence | boundary | 8/8 | 6 | 3/6（8） | 3/8 | 6 |
| minimax/minimax-m2.5 | evidence | complex | 8/8 | 6 | 2/6（8） | 3/8 | 2 |
| minimax/minimax-m2.5 | evidence | routine | 8/8 | 8 | 6/8（8） | 0/8 | 2 |
| minimax/minimax-m2.5 | history | all | 24/24 | 18 | 14/18（24） | 4/21 | 2 |
| minimax/minimax-m2.5 | history | boundary | 8/8 | 7 | 5/7（8） | 1/7 | 2 |
| minimax/minimax-m2.5 | history | complex | 8/8 | 5 | 3/5（8） | 2/6 | 0 |
| minimax/minimax-m2.5 | history | routine | 8/8 | 6 | 6/6（8） | 1/8 | 0 |
| qwen/qwen3.5-27b | all | all | 72/72 | 53 | 40/53（72） | 11/66 | 15 |
| qwen/qwen3.5-27b | all | boundary | 24/24 | 16 | 12/16（24） | 2/21 | 4 |
| qwen/qwen3.5-27b | all | complex | 24/24 | 20 | 17/20（24） | 4/23 | 4 |
| qwen/qwen3.5-27b | all | routine | 24/24 | 17 | 11/17（24） | 5/22 | 7 |
| qwen/qwen3.5-27b | consultation | all | 24/24 | 16 | 12/16（24） | 6/22 | 6 |
| qwen/qwen3.5-27b | consultation | boundary | 8/8 | 5 | 5/5（8） | 0/8 | 0 |
| qwen/qwen3.5-27b | consultation | complex | 8/8 | 7 | 5/7（8） | 3/8 | 3 |
| qwen/qwen3.5-27b | consultation | routine | 8/8 | 4 | 2/4（8） | 3/6 | 3 |
| qwen/qwen3.5-27b | evidence | all | 24/24 | 18 | 12/18（24） | 5/23 | 7 |
| qwen/qwen3.5-27b | evidence | boundary | 8/8 | 6 | 4/6（8） | 2/8 | 3 |
| qwen/qwen3.5-27b | evidence | complex | 8/8 | 6 | 5/6（8） | 1/7 | 1 |
| qwen/qwen3.5-27b | evidence | routine | 8/8 | 6 | 3/6（8） | 2/8 | 3 |
| qwen/qwen3.5-27b | history | all | 24/24 | 19 | 16/19（24） | 0/21 | 2 |
| qwen/qwen3.5-27b | history | boundary | 8/8 | 5 | 3/5（8） | 0/5 | 1 |
| qwen/qwen3.5-27b | history | complex | 8/8 | 7 | 7/7（8） | 0/8 | 0 |
| qwen/qwen3.5-27b | history | routine | 8/8 | 7 | 6/7（8） | 0/8 | 1 |

| 模型 | 首次尝试完成/已尝试（计划） | 尝试总数 | 有重试的运行 | 恢复后完成 |
|---|---:|---:|---:|---:|
| gemini-3.5-flash | 68/72（72） | 72 | 0 | 0 |
| minimax/minimax-m2.5 | 69/72（72） | 72 | 0 | 0 |
| qwen/qwen3.5-27b | 69/72（72） | 72 | 0 | 0 |

## 重复运行（稳定性补充，不并入主样本）

| 模型 | 领域 | 难度 | 已执行/计划 | 完成评分 | 全适用项通过/评分（计划） | 关键失败/已评运行 | 分歧项 |
|---|---|---|---:|---:|---:|---:|---:|
| gemini-3.5-flash | all | all | 28/36 | 17 | 12/17（36） | 3/21 | 5 |
| gemini-3.5-flash | all | boundary | 9/12 | 5 | 3/5（12） | 3/7 | 2 |
| gemini-3.5-flash | all | complex | 9/12 | 5 | 4/5（12） | 0/5 | 1 |
| gemini-3.5-flash | all | routine | 10/12 | 7 | 5/7（12） | 0/9 | 2 |
| gemini-3.5-flash | consultation | all | 12/12 | 8 | 5/8（12） | 1/9 | 2 |
| gemini-3.5-flash | consultation | boundary | 4/4 | 3 | 2/3（4） | 1/4 | 0 |
| gemini-3.5-flash | consultation | complex | 4/4 | 2 | 2/2（4） | 0/2 | 0 |
| gemini-3.5-flash | consultation | routine | 4/4 | 3 | 1/3（4） | 0/3 | 2 |
| gemini-3.5-flash | evidence | all | 10/12 | 8 | 6/8（12） | 2/10 | 3 |
| gemini-3.5-flash | evidence | boundary | 3/4 | 2 | 1/2（4） | 2/3 | 2 |
| gemini-3.5-flash | evidence | complex | 3/4 | 3 | 2/3（4） | 0/3 | 1 |
| gemini-3.5-flash | evidence | routine | 4/4 | 3 | 3/3（4） | 0/4 | 0 |
| gemini-3.5-flash | history | all | 6/12 | 1 | 1/1（12） | 0/2 | 0 |
| gemini-3.5-flash | history | boundary | 2/4 | 0 | 0/0（4） | 0/0 | 0 |
| gemini-3.5-flash | history | complex | 2/4 | 0 | 0/0（4） | 0/0 | 0 |
| gemini-3.5-flash | history | routine | 2/4 | 1 | 1/1（4） | 0/2 | 0 |
| minimax/minimax-m2.5 | all | all | 28/36 | 22 | 12/22（36） | 7/26 | 14 |
| minimax/minimax-m2.5 | all | boundary | 9/12 | 6 | 3/6（12） | 4/8 | 5 |
| minimax/minimax-m2.5 | all | complex | 9/12 | 7 | 4/7（12） | 1/8 | 3 |
| minimax/minimax-m2.5 | all | routine | 10/12 | 9 | 5/9（12） | 2/10 | 6 |
| minimax/minimax-m2.5 | consultation | all | 12/12 | 10 | 3/10（12） | 4/12 | 9 |
| minimax/minimax-m2.5 | consultation | boundary | 4/4 | 3 | 1/3（4） | 2/4 | 2 |
| minimax/minimax-m2.5 | consultation | complex | 4/4 | 4 | 2/4（4） | 0/4 | 2 |
| minimax/minimax-m2.5 | consultation | routine | 4/4 | 3 | 0/3（4） | 2/4 | 5 |
| minimax/minimax-m2.5 | evidence | all | 10/12 | 8 | 5/8（12） | 3/10 | 5 |
| minimax/minimax-m2.5 | evidence | boundary | 3/4 | 2 | 1/2（4） | 2/3 | 3 |
| minimax/minimax-m2.5 | evidence | complex | 3/4 | 2 | 1/2（4） | 1/3 | 1 |
| minimax/minimax-m2.5 | evidence | routine | 4/4 | 4 | 3/4（4） | 0/4 | 1 |
| minimax/minimax-m2.5 | history | all | 6/12 | 4 | 4/4（12） | 0/4 | 0 |
| minimax/minimax-m2.5 | history | boundary | 2/4 | 1 | 1/1（4） | 0/1 | 0 |
| minimax/minimax-m2.5 | history | complex | 2/4 | 1 | 1/1（4） | 0/1 | 0 |
| minimax/minimax-m2.5 | history | routine | 2/4 | 2 | 2/2（4） | 0/2 | 0 |
| qwen/qwen3.5-27b | all | all | 28/36 | 18 | 13/18（36） | 6/24 | 5 |
| qwen/qwen3.5-27b | all | boundary | 9/12 | 5 | 4/5（12） | 3/7 | 1 |
| qwen/qwen3.5-27b | all | complex | 9/12 | 5 | 5/5（12） | 0/7 | 0 |
| qwen/qwen3.5-27b | all | routine | 10/12 | 8 | 4/8（12） | 3/10 | 4 |
| qwen/qwen3.5-27b | consultation | all | 12/12 | 7 | 5/7（12） | 3/12 | 2 |
| qwen/qwen3.5-27b | consultation | boundary | 4/4 | 3 | 2/3（4） | 2/4 | 1 |
| qwen/qwen3.5-27b | consultation | complex | 4/4 | 2 | 2/2（4） | 0/4 | 0 |
| qwen/qwen3.5-27b | consultation | routine | 4/4 | 2 | 1/2（4） | 1/4 | 1 |
| qwen/qwen3.5-27b | evidence | all | 10/12 | 8 | 5/8（12） | 3/9 | 3 |
| qwen/qwen3.5-27b | evidence | boundary | 3/4 | 1 | 1/1（4） | 1/2 | 0 |
| qwen/qwen3.5-27b | evidence | complex | 3/4 | 3 | 3/3（4） | 0/3 | 0 |
| qwen/qwen3.5-27b | evidence | routine | 4/4 | 4 | 1/4（4） | 2/4 | 3 |
| qwen/qwen3.5-27b | history | all | 6/12 | 3 | 3/3（12） | 0/3 | 0 |
| qwen/qwen3.5-27b | history | boundary | 2/4 | 1 | 1/1（4） | 0/1 | 0 |
| qwen/qwen3.5-27b | history | complex | 2/4 | 0 | 0/0（4） | 0/0 | 0 |
| qwen/qwen3.5-27b | history | routine | 2/4 | 2 | 2/2（4） | 0/2 | 0 |

| 模型 | 首次尝试完成/已尝试（计划） | 尝试总数 | 有重试的运行 | 恢复后完成 |
|---|---:|---:|---:|---:|
| gemini-3.5-flash | 21/28（36） | 28 | 0 | 0 |
| minimax/minimax-m2.5 | 26/28（36） | 28 | 0 | 0 |
| qwen/qwen3.5-27b | 24/28（36） | 28 | 0 | 0 |

## 可追溯指标与限制

[完整指标与分层分母](agent/holdout_v1_live_20260908/holdout_metrics.json) 包含逐检查点四类裁决计数、执行/评分状态、重复稳定性、Token、工具与检索次数、延迟分布、服务环境和原始记录位置。
费用合并 ledgers/ 各进程账本，以去重 request_id 为依据，区分服务端报告、Token 估算和未知费用。目标调用、患者模拟与评审成本按 role 分开，不混入目标模型成本；Mem0 货币费用未由服务暴露。
首次患者可见安全提示时间尚未作语义定位，报告留空；内部子 Agent 文本不能替代患者已看到的提示。
关键失败、评审分歧和预先随机抽取的通过样本仍需独立复核；本自动报告不声称复核已完成。
场景家族是独立单位；重复稳定仅描述相同场景的重跑表现。本报告不计算将轮次当独立样本的置信区间，也不认证罕见安全事故率。


本页为 2026-09-08 发布快照，自动双评尚未完整结束。可用状态证据：[技术边界审计](agent/holdout_v1_live_20260908/technical_boundary_audit.json)、[RAG 上下文审计](agent/holdout_v1_live_20260908/rag_context_audit.json)、[运行与延迟汇总](agent/holdout_v1_live_20260908/telemetry_summary.json)。

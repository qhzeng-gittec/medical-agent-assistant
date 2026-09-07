# 医疗问诊 Agent 测评结果（开发回归版）

日期：2026-09-07。真实调用 OpenRouter；全部病例为模拟患者，不含真实患者个人信息。
这份报告验证工程行为和小范围问诊表现，不是临床认证、诊断准确率或上线许可。

**当前状态：48 个主测任务已完整执行；自动双评审只有 16/96 条有效评分，剩余 80 条待补。** 评审阶段先遇到并发额度预占 402；随后因账户额度不足暂停付费请求。当前不发布整体医疗质量通过率，也不按部分评分给模型排名。

## 1. 开会先讲什么

这次不是让模型做医学选择题，而是把患者放进真实 Supervisor → 专业 Agent → 工具 → 回复的链路里。既看有没有问对问题和处理危险信号，也看工具调用、记忆读写、证据使用、错误恢复以及实际耗时。
共 24 个场景、2 款可用被测模型，形成 48 个主测任务。第三款 Gemini 在预检返回服务商 403；没有替换型号或把拒绝访问算成能力失败。

## 2. 测了什么，没测什么

| 层次 | 真实执行内容 | 不应夸大的边界 |
|---|---|---|
| 工程回归 | 原项目自动测试，以及 8 个 RAG 缓存/正文引用合约场景 | 通过只证明已写断言，不证明医学正确 |
| 真实 RAG | OpenRouter qwen/qwen3-embedding-8b，4096 维；本地保存向量并做余弦检索 | 没有本机 embedding 推理；不是完整 Milvus 部署压测 |
| 问诊主流程 | 追问、急症、用药边界、续问、更正、用户隔离、档案持久化、异常与注入 | 24 个开发场景，不是独立测试集或真实临床患者研究 |
| 长期事实 | 实际 PatientProfileStore 文件读写及新会话读取 | Mem0 Platform 无凭据，本轮未连通；不能声称长期事件记忆通过 |
| 专业模型 | 同一次任务的总控和子 Agent 都使用对应被测通用模型 | 没有微调医疗模型服务端点，未验证微调收益或异构模型分工 |
| 外网深度研究 | 如被调用返回明确不可用错误 | 未配置外网研究服务，不伪造成功结果 |

知识库来自 NHS 患者信息和 CDC 成人门诊指导页面的 15 个中文摘要块。摘要由本次测评以 AI 辅助整理，未经过独立医生审核；字段中‘人工整理’不代表医生编写或审核。NHS 患者说明也不等同于正式临床指南。

方法参考：[OpenRouter embedding 接口](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request)、[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)；查询加任务说明，文档不加。[逐轨迹评估思路](https://developers.openai.com/api/docs/guides/trace-grading)用于区分最终答复和工具过程。

## 3. 真实测量结果

| 指标 | MiniMax M2.5 | Qwen3.5-27B |
|---|---:|---:|
| 完整走到任务结尾 / 24 | 24 | 24 |
| 存在子 Agent 失败的任务 | 0 | 3 |
| 已完成的患者轮次 | 41 | 42 |
| 每轮平均耗时（秒） | 29.4 | 64.0 |
| 每轮 P95 耗时（秒） | 72.7 | 158.3 |
| 主链路模型请求数 | 142 | 175 |
| 实际检索后端执行数 | 71 | 104 |
| 工具结果中的新正文块 / 引用另见下一行 | 29 | 38 |
| 工具结果中的正文引用块 | 2 | 5 |
| 发往模型的上下文配对/引用违规 | 0 | 0 |
| 收到的空结果/工具协议异常 | 0 | 3 |
| 两位评审均成功评分的任务 | 3 | 4 |
| 主链路服务商回报费用（美元） | 0.06956 | 0.21102 |

‘走到结尾’不等于答案合格；子 Agent 失败后总控仍可能生成答复。耗时为非流式、每个已完成用户轮次到最终回复的墙钟时间，不包括模拟患者/评审用时，也不是首 token 延迟；4 个独立患者任务并发，云服务抖动和超时上限会影响统计。未完成轮次不能被当成 0 秒。

单独看已预置明确危险信号的 5 类场景（07–10、24）：记录了 10 次目标轮回复，其中 10 次先等待子 Agent 链路；最终回复耗时范围 21.0–90.6 秒。这是流程时延事实，不等于这些答复经过了医生的安全审核。

工程测试输出：`62 passed in 1.05s`。RAG 小检索集：10 个库内问题，Top-1 命中率 100%，Top-3 命中率 100%。这是与 15 个精选块配套的冒烟集，不是大规模泛化成绩。

仅在目前已完成的部分评分中，两位自动评审在 35 个可对应条目中有 5 个判定不一致。样本按执行顺序缺失，不能代表全部场景。没有把模型评审伪装成人类医生结论；也不依据这么小的样本给两款模型排临床优劣。详见逐项理由和原始对话。

## 4. 测试过程中发现的问题如何解释

详见同目录《问题复盘与下一轮计划.md》。关键点不是‘多 Agent 能否运行’，而是：追问前是否过早宣布低风险、提示词是否强迫编造概率、风险规则能否理解否定、急救提示是否被工具链阻塞、自然语言档案更新是否真正落盘、证据是否只是检索到了却没有传给最终答复。

重复检索与重复正文必须分开：本轮使用原项目的 worker 内完全相同参数缓存；跨 Agent 不共享中间检索。正文引用检查只证明可见上下文里复用了同一块，不能直接得出减少了多少 API 调用。summary 中的正文重新展开字符差只是一条轨迹的离线估计，不是与无去重版本配对实测的 token 节省率。

## 5. 测评器也接受审计

第一轮患者事实选择器的 2048 输出额度发生截断；因此将该模拟器额度统一升到 4096，把全部 6 类可交互场景、两款被测模型一起重跑为 r4，不按旧分数挑选重跑。其他 18 类固定脚本场景使用 r3。被测 Agent 仍统一 temperature=0.2、reasoning effort=low、单次输出上限 2048，专业 Agent 工具预算 2，总控轮数上限 4。
这里的 low 是统一发送的请求参数，不代表不同供应商/模型采用了完全相等的内部推理量；本轮可以比较这套运行配置的表现，不能宣称已经严格隔离所有模型与供应商差异。
升到 4096 后个别患者选择仍截断，随后只对这些测评器断点关闭患者模型的 reasoning，再继续未完成的患者补充。之前的医疗回答、用户事实、短期历史和档案原样恢复，没有重抽前一轮医疗回答。恢复结果以 _resumed 单独保存，原失败仍保留。
初版评审自行摘抄引文时出现原文对不上，均记录为 judge_error。v2 改成选答案行 ID，程序从原记录提取证据；计划让 MiniMax 与 Qwen 两位评审都评两款模型，目前因额度不足只完成部分。原始失败和旧评分没有删除。

## 6. 费用

- provider_reported：$0.871402
- unknown_cost_reserved：$1.247060

provider_reported 是 API 回报的费用，不是独立账单核对；estimated 是按回报 token 和目录价格估算；unknown/reserved 是失败请求没有费用字段时保留的预算占位，不能说已实际扣款。总额包含连通性预检、embedding、患者模拟、旧版失败、重跑和自动评审。两次额外 403 诊断请求无 usage，未据此断言免费。

## 7. 如何继续迭代

固定此版场景、来源和产品源码哈希，把真实失败加入回归。改一个问题后，在同一场景、相同推理预算和同一 RAG 快照下配对重跑，至少保留多次重复，再报告通过率、耗时、token 的差值。当前结果是现状基线，没有旧架构配对数据，不能编造‘提升 X%’。
扩展时按病例家族划分 dev / test，冻结 holdout，并请医生独立审核病例事实、风险边界和评分标准；新增更长多轮、模糊口语、否定/更正、家属代问、年龄与孕期、药物相互作用，以及 Mem0 事件过期与隔离测试。

## 附：每个任务的结果入口

| 病例 | 被测模型 | 执行 | 有效评审数量 / 2 | 原始记录 |
|---|---|---|---|---|
| CONSULT-01 | minimax/minimax-m2.5 | completed | 2/2 | 对话 JSON（`CONSULT-01__minimax_minimax-m2.5__r4.json`，原始记录未发布） |
| CONSULT-01 | qwen/qwen3.5-27b | completed | 2/2 | 对话 JSON（`CONSULT-01__qwen_qwen3.5-27b__r4.json`，原始记录未发布） |
| CONSULT-02 | minimax/minimax-m2.5 | completed | 2/2 | 对话 JSON（`CONSULT-02__minimax_minimax-m2.5__r4.json`，原始记录未发布） |
| CONSULT-02 | qwen/qwen3.5-27b | completed | 2/2 | 对话 JSON（`CONSULT-02__qwen_qwen3.5-27b__r4.json`，原始记录未发布） |
| CONSULT-03 | minimax/minimax-m2.5 | completed | 1/2 | 对话 JSON（`CONSULT-03__minimax_minimax-m2.5__r4.json`，原始记录未发布） |
| CONSULT-03 | qwen/qwen3.5-27b | completed | 2/2 | 对话 JSON（`CONSULT-03__qwen_qwen3.5-27b__r4.json`，原始记录未发布） |
| CONSULT-04 | minimax/minimax-m2.5 | completed | 2/2 | 对话 JSON（`CONSULT-04__minimax_minimax-m2.5__r4.json`，原始记录未发布） |
| CONSULT-04 | qwen/qwen3.5-27b | completed | 2/2 | 对话 JSON（`CONSULT-04__qwen_qwen3.5-27b__r4.json`，原始记录未发布） |
| CONSULT-05 | minimax/minimax-m2.5 | completed | 1/2 | 对话 JSON（`CONSULT-05__minimax_minimax-m2.5__r4.json`，原始记录未发布） |
| CONSULT-05 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-05__qwen_qwen3.5-27b__r4.json`，原始记录未发布） |
| CONSULT-06 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-06__minimax_minimax-m2.5__r4_resumed.json`，原始记录未发布） |
| CONSULT-06 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-06__qwen_qwen3.5-27b__r4.json`，原始记录未发布） |
| CONSULT-07 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-07__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-07 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-07__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-08 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-08__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-08 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-08__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-09 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-09__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-09 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-09__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-10 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-10__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-10 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-10__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-11 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-11__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-11 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-11__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-12 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-12__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-12 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-12__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-13 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-13__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-13 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-13__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-14 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-14__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-14 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-14__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-15 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-15__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-15 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-15__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-16 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-16__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-16 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-16__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-17 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-17__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-17 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-17__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-18 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-18__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-18 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-18__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-19 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-19__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-19 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-19__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-20 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-20__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-20 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-20__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-21 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-21__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-21 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-21__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-22 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-22__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-22 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-22__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-23 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-23__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-23 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-23__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |
| CONSULT-24 | minimax/minimax-m2.5 | completed | 0/2 | 对话 JSON（`CONSULT-24__minimax_minimax-m2.5__r3.json`，原始记录未发布） |
| CONSULT-24 | qwen/qwen3.5-27b | completed | 0/2 | 对话 JSON（`CONSULT-24__qwen_qwen3.5-27b__r3.json`，原始记录未发布） |

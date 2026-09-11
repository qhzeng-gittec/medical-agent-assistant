# MediX 多智能体医疗助手

这是医疗助手的运行模块：接收用户问题和可选图像，由一个总控 Agent 分配任务，专业 Agent 调用检索、图像分析等工具，最后汇总为面向用户的回答。跨会话记忆用于找回用户此前提供的信息。

三个专业角色分别是：Consultation（咨询与健康资料检索）、Diagnostic（图像分析）、Research（文献与网络研究）。总控由 `MedicalSupervisorAgent` 实现；核心代码位于 `swarm/`、`agents/`、`core/` 和 `memory/`，工具实现位于 `.claude/skills/`。

想先了解实际效果，可从[测评总览](../results/README.md)阅读任务表现、架构对照和检索/记忆实验；本页主要说明执行流程、配置和运行方法。

## 执行流程与上下文

一次用户提问到最终回答是一次请求；同一 `session_id` 下的多次问答组成会话。每次请求内部，总控可以多次调用模型、执行工具、读取结果，默认最多 4 个内部轮次，耗尽后另作无工具总结。每个专业子任务内部默认最多 5 个模型轮次。内部轮次不会自动产生一条新的用户消息。

总控先读取患者档案、最近最多 5 轮且约 8,000 字符的对话，并在长期记忆启用时自动召回最多 3 条相关历史。若当前资料不能覆盖用户要求，模型可以在本次请求的后续内部轮次继续调用 `search_patient_history`，每次默认最多 5 条、可指定 1～10 条。例如用户要求整理疾病、过敏、作息和运动限制，首次仅召回前两项，就可以针对后两项补查。是否存在缺口、使用什么查询由模型判断；仍找不到时应说明未知。此路径使用 Mem0 的 embedding 和向量存储接口，按用户及应用过滤。

子任务获得具体 `task`、原始用户问题和以下背景：

| 字段 | 内容与边界 |
| --- | --- |
| `user_context` | 患者档案、最近对话、历史召回及调用方背景；当前没有按专业精细筛选 |
| `prior_agent_findings` | 本次请求此前所有成功子任务的交付，排除原始 `evidence`；不是上一轮用户对话，也不按依赖关系筛选 |
| 子任务消息与缓存 | 每次调用重新建立，互不继承完整工具过程；同轮任务看不到彼此结果 |

相互独立的任务可以同轮并行，包括同一专业的不同任务；同轮同专业且参数相同的重复提交会被拒绝。需要先读另一个任务结论时，由总控分轮委派；代码没有预设依赖图。同轮档案更新和历史补查先完成，再启动专业任务。

子任务返回 `answer` 和实际工具召回的 `evidence`。总控收到最多 8,000 字符答案及累计最多 8,000 字符证据正文，附来源元信息与截断标记；证据按收集顺序截取，没有实现逐结论的引用对齐。总控有原文可核对，并不保证模型一定完成有效核验。[总控代码](swarm/supervisor_agent.py) · [子任务循环](core/agent_loop.py)

## 当前工具

总控调用三个专业 Agent，并在身份和配置允许时更新患者档案、补查历史。六个业务工具按 2 / 1 / 3 分配：

| Agent | 工具 | 实际执行 |
| --- | --- | --- |
| 诊断 | `analyze_symptoms` | 检索风险与症状分析候选资料，最多 3 条；风险判断由模型完成 |
| 诊断 | `disease_code` | 检索疾病分类资料并提取 ICD-10 编码 |
| 咨询 | `recommend_lifestyle` | 检索生活方式类型资料 |
| 研究 | `search_knowledge` | 不限制资料类型的知识检索 |
| 研究 | `clinical_guideline` | 检索指南类型，当前包装第一条合格结果 |
| 研究 | `deep_research` | 请求 Tavily，返回网页正文或摘要与来源链接 |

旧 `assess_risk` 与症状工具使用相同检索参数，已合并到 `analyze_symptoms`。资料默认共用 Milvus 的 `medical_knowledge` collection；`lifestyle`、`disease_classification`、`clinical_guideline` 是资料类型过滤，不是三个独立数据库。

本项目的 Skill 由 `SKILL.md` 元数据和 Python 函数组成。注册器生成工具描述及参数结构，模型通过正式工具调用选择函数；当前不按需加载完整技能正文。

`deep_research(query, max_iterations=5)` 的参数控制搜索规模，允许 1～5，每单位最多 3 个来源，即默认最多 15 个；它不是研究轮数，一次工具调用只发出一次搜索请求。每个来源最多保留 6,000 字符，并标明正文或摘要与截断情况。继续检索由研究 Agent 的循环决定；扩大返回量尚无新的质量对照。

## 安装和运行

建议使用独立 Python 3.11+ 环境。以下 PowerShell 命令从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r medix-agent-swarm/requirements.txt -r requirements-dev.txt
Copy-Item config.example.py config.py
$env:LLM_API_KEY = "你的 API key"
$env:LLM_MODEL = "你的模型 ID"
$env:LLM_BASE_URL = "你的 OpenAI 兼容 API 地址"
python medix-agent-swarm/main.py
```

Linux/macOS 使用 `source .venv/bin/activate`、`cp config.example.py config.py` 和 `export` 设置相同环境变量。

`config.py` 已被 Git 忽略。可选长期记忆使用 Mem0 OSS 和本地 Qdrant，通过 `OPENROUTER_API_KEY` 调用 Qwen3.5-27B 抽取及 Qwen3-Embedding-8B 嵌入；未配置该变量时不启用长期记忆。数据库默认保存在 `medix-agent-swarm/.mem0/`，可用 `MEM0_LOCAL_PATH` 指定其他目录。

```powershell
$env:OPENROUTER_API_KEY = "你的 OpenRouter API key"
python medix-agent-swarm/main.py
```

记忆按用户和应用隔离，跨会话检索保留来源与时间；入口在请求结束时关闭存储连接。数据保存在本机，抽取和嵌入请求会发送到配置的 API。

网络搜索另需设置 `TAVILY_API_KEY`。只在调用 `deep_research` 时请求 Tavily；未配置或服务失败会明确返回工具错误。来源内容交给研究 Agent 核对，不请求额外生成式答案。

## 知识库

生产检索实现使用 Milvus 与 sentence-transformers，需要另行准备对应服务/运行环境与嵌入模型。`knowledge/data/documents/` 只附带一条技术演示文本，不是医学指南或真实诊疗资料。加入自己有权使用的资料后，在本目录运行：

```powershell
python knowledge/scripts/import_hardcoded_data.py
```

初始化程序读取文档并构建索引；数据库不随 Git 提交。运行问诊时若触发检索，必须先准备相应后端。离线测试使用桩对象，不会因此证明真实 Milvus 或 Mem0 已部署成功。

## 离线测试

从仓库根目录执行，先按上文复制配置模板：

```powershell
python -m pytest medix-agent-swarm/tests medix-agent-swarm/evals/test_campaign.py medix-agent-swarm/evals/test_dataset_tools.py -q
python medix-agent-swarm/evals/dataset_tools.py
python medix-agent-swarm/evals/run_components.py
```

测试验证路由、档案隔离、上下文预算、证据复用及评测协议；不调用付费模型。

## 真实模型评测

`evals/campaign_v1/` 包含 24 个开发场景与 15 个来源摘要。`cases.private.jsonl` 中的评分规则和未披露患者事实由运行器管理，不能整体发送给被测模型。旧开发集 `evals/datasets/medix_dev_v0_1/` 保留用于溯源与组件测试；其 `provenance/LICENSE-CMB.txt` 是上游 CMB 许可证。

在本目录配置 `OPENROUTER_API_KEY` 后，可新建一个实验输出目录：

```powershell
python evals/campaign_run.py --output evals/results/my_run --repetition 1 --concurrency 1
```

当前整体验收运行器使用 MiniMax M2.5、Qwen3.5-27B 和 GPT-5.5。前两者与嵌入通过 OpenRouter 调用，GPT-5.5 通过本机已登录 ChatGPT 账号的 Codex CLI 调用，需要 `codex` 在 PATH 中可用。调用记录区分 API 费用与 ChatGPT 账号额度。`campaign_resume.py`、`campaign_report.py` 等保留原实验的恢复和汇总逻辑，含固定批次选择；新评测使用新的输出目录。已发布报告中的历史模型与数据按各自冻结协议保留。

9 月 9 日冻结版本的 72 场景结果、六项关键设计和逐场证据见 [Agent 整体测评](../results/agent-current-2026-09-09/README.md)。本次 9 月 10 日更新未重跑完整模型测评。运行器将新实验的轨迹和评分保存到指定输出目录。

## 离线演示

在本目录运行 `python -m examples.context_growth`，可查看真实编排代码收到的消息与证据复用轨迹。模型回复和知识库内容均使用固定模拟数据，不发送付费请求。回归测试位于 `tests/`。

## 关键设计与评测

[冻结 Agent 设计与整体测评](../results/agent-current-2026-09-09/README.md)公开 72 场历史回归中的逐轮回答、工具轨迹及双评结果。更大规模的组件实验包括 [800 条 RAG 查询](../results/hybrid-grounding-2026-09-08/README.md)和 [120 个 Mem0 场景](../results/component-benchmark-2026-09-08/memory/README.md)。组件召回、最终回答、总控主动补查和多 Agent 架构收益是不同问题，不能互相替代。

Supervisor 按依赖调度专业 Worker，核对其返回的真实检索摘录；正文中的伪工具调用不会当作已执行结果。研究回答保留来源条件，避免将缺失依据补写成已验证结论。同一 Worker 内继续复用相同请求和文档正文。

新写入的 Mem0 记忆附带最多 4,000 个字符的用户原话及截断标记，保留事件时间与咨询时间，系统保存时间不作为事件日期。已有记忆不回填原话。

Worker 的 `max_tool_calls` 默认 `None`（不限制调用次数），可在 Agent 配置中设置整数上限；执行层继续强制已配置的上限，轮数及超时边界保持有效。历史工具预算审计对应其原有配置。

9 月 9 日记录有 104 项 Agent 测试和 48 项评测运行器测试通过；本次更新通过 81 项针对性测试，检查同专业任务并行、角色工具集合、搜索边界、协议、上下文、证据交付和缓存。上述测试使用确定性网络输出，不代表真实后端质量验收。

[关键测评](../results/README.md)按调度效率、检索覆盖和模型训练分别说明模型角色与实测结果。默认问诊模型通过 `LLM_MODEL` 配置；长期记忆的抽取与嵌入配置见安装说明。

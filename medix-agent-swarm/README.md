# MediX 多智能体医疗助手

运行时由 `MedicalSupervisorAgent` 调度 Consultation、Diagnostic、Research 三类专业 Agent。核心代码位于 `swarm/`、`agents/`、`core/` 和 `memory/`，工具实现位于 `.claude/skills/`。

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

关键设计的数据与测量方式见 [Agent 测评报告](../results/2026-09-08/agent_improvements.md)，完整场景验收见 [测评方法与评分表](../results/2026-09-08/agent_holdout.md)。运行器将新实验的轨迹和评分保存到指定输出目录。

## 离线演示

在本目录运行 `python -m examples.context_growth`，可查看真实编排代码收到的消息与证据复用轨迹。模型回复和知识库内容均使用固定模拟数据，不发送付费请求。回归测试位于 `tests/`。

## 关键设计与评测

系统采用独立 Worker 条件并行、患者档案/近期对话/可检索历史的分层记忆、主动补查、RAG 多候选、同一 Worker 内证据复用和执行层工具预算控制。

[关键设计报告](../results/2026-09-08/agent_improvements.md)逐项给出解决的问题、实测数据、对照条件与适用范围；[整体验收](../results/2026-09-08/agent_holdout.md)说明 72 个合成场景如何构造、逐轮检查和双评；[架构对照](../results/2026-09-08/architecture.md)分别报告子任务并行收益及完整流程代价。

新增 [RAG / Mem0 独立组件测评](../results/component-benchmark-2026-09-08/README.md)：RAG 在 1,016 篇语料、400 条查询上测试，冻结测试集 Top 1 → Top 3 的完整目标来源覆盖为 **183/272 → 249/272**；Mem0 在 120 个场景上测试，测试集 Top 3 → Top 10 的全部事实支持为 **176/192 → 184/192**。数据、脚本和原始记录均可直接下载。

当前总控结合完整语义选择分派与并行顺序，症状工具返回候选证据，专业 Agent 负责分析并交付实际来源。执行层校验工具可用性、用户作用域和调用预算。既有固定子任务并行对照记录 **59.5%** 的耗时缩短，原始配置与记录见 [结果索引](../results/2026-09-08/README.md)。

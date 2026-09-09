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

当前系统回归与记忆修复见 [设计与测评报告](../results/memory-2026-09-09/README.md)，其他关键实验见 [精选测评](../results/README.md)。运行器将新实验的轨迹和评分保存到指定输出目录。

## 离线演示

在本目录运行 `python -m examples.context_growth`，可查看真实编排代码收到的消息与证据复用轨迹。模型回复和知识库内容均使用固定模拟数据，不发送付费请求。回归测试位于 `tests/`。

## 关键设计与评测

Supervisor 按依赖调度专业 Worker，核对其返回的真实检索摘录；正文中的伪工具调用不会当作已执行结果。研究回答保留来源条件，避免将缺失依据补写成已验证结论。同一 Worker 内继续复用相同请求和文档正文。

新写入的 Mem0 记忆附带最多 4,000 个字符的用户原话及截断标记，保留事件时间与咨询时间，系统保存时间不作为事件日期。已有记忆不回填原话。

Worker 的 `max_tool_calls` 默认 `None`（不限制调用次数），可在 Agent 配置中设置整数上限；执行层继续强制已配置的上限，轮数及超时边界保持有效。历史工具预算审计对应其原有配置。

[系统回归与记忆修复](../results/memory-2026-09-09/README.md)给出当前代码对应的配对结果、分母、退步项和原始证据。[混合检索与回答测评](../results/hybrid-grounding-2026-09-08/README.md)分别测量检索覆盖与回答质量；混合检索为实验适配器，产品默认仍用 Milvus。[架构对照](../results/2026-09-08/architecture.md)报告子任务并行收益和完整流程代价。

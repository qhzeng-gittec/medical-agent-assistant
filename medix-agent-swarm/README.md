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

`config.py` 已被 Git 忽略。Mem0 通过可选 `MEM0_API_KEY` 配置，未配置时不启用远端记忆。

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

这会调用付费外部 API。运行器里的模型 ID 是实验配置，使用前确认账号可访问。`campaign_resume.py`、`campaign_report.py` 等保留原实验的恢复和汇总逻辑，含固定批次选择，不能直接当作任意新实验的通用报告生成器。

实验设置、结果和已知问题见 [开发测评报告](reports/医疗问诊Agent测评报告.md)。运行器将新实验的轨迹和评分保存到指定输出目录。

## 离线演示

在本目录运行 `python -m examples.context_growth`，可查看真实编排代码收到的消息与证据复用轨迹。模型回复和知识库内容均使用固定模拟数据，不发送付费请求。回归测试位于 `tests/`。

## 2026-09-08 更新与实测

档案更新改为 Supervisor 通过 `update_patient_profile` 提议、执行层校验原话证据和用户作用域；新增 `search_patient_history` 主动补查。Mem0 写入保留真实角色和来源；生活方式检索返回 Top 3 候选并明确空结果；Worker 背景只注入一次。Evidence Store 的正文复用仍限于同一 Worker 迭代，不保证跨 Worker 检索去重。

[8 场景改前/改后](../results/2026-09-08/agent_improvements.md)、[72 场景新执行](../results/2026-09-08/agent_holdout.md)、[单/多 Agent 及串并行](../results/2026-09-08/architecture.md)分别保留协议、分母和失败。完整架构未证明更快、更准；新版留出中仍发现虚假完成声明、档案更正不完整、Mem0 metadata 超限和证据归因错误。

开发集、模型/服务采样与受控记录池不互相冒充。发布后的这些场景已公开，后续最终验收须另建未参与调试的独立数据。

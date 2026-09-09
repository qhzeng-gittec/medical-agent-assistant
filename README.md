# Medical Agent Assistant

医疗多智能体助手与 Qwen3.5-2B 医疗模型训练实验。支持多轮问诊、专业 Agent 调度、患者档案和检索证据管理，并提供 CPT、LoRA SFT、GSPO 训练与可追溯评测。

更新于 **2026-09-09**：同步证据核对、工具调用协议和跨会话记忆改进；补充迭代后保留的关键设计、72 场景 Agent 整体测评，以及检索与模型训练结果。

[Agent 使用说明](medix-agent-swarm/README.md) · [训练与推理](MediX-R1/README.md) · [Agent 关键设计](#agent-关键设计) · [Agent 整体测评](#agent-整体测评) · [全部精选测评](results/README.md) · [模型与数据](https://huggingface.co/collections/starttoshow/medix-medical-sft-and-gspo-6a9e6f31b80642f4ba8b6f28)

## 功能

| 模块 | 功能 |
| --- | --- |
| 医疗 Agent | Supervisor 根据问题调用问诊、诊断、研究 Agent，汇总工具结果和来源 |
| 记忆与检索 | 模型提议档案更新，代码校验用户和原话证据；近期对话预算、可选 Mem0 与主动历史补查；RAG 候选和正文复用 |
| 模型训练 | Qwen3.5-2B 的 Attention / Attention+FFN LoRA、VQA GSPO，以及 CPT→SFT→知识/病例 GSPO 对照链 |
| 评测 | 逐轮回答/档案/工具状态核对、逐项双评、串并行对照、答案与解释分评、配对置信区间 |

项目包含两条可独立运行的工程链路：通过 OpenAI 兼容 API 驱动的医疗 Agent，以及 Qwen3.5-2B 的 LoRA 训练与推理。

## Agent 关键设计

项目迭代围绕总控如何分工、跨轮状态怎样更新、历史如何回查以及证据怎样交付展开。当前保留六项核心设计：

| 设计 | 当前实现 |
| --- | --- |
| **按语义与依赖调度** | 总控根据问题和已有结果选择专业 Agent；独立任务同轮并行，有依赖的任务分轮执行，各 Worker 只获得职责内工具 |
| **分层记忆与上下文预算** | 患者档案保存稳定事实，近期对话按完整轮次与字符预算保留，Mem0 提供跨会话事件检索 |
| **有原话依据的档案更新** | 模型提议新增、更正、否定和停用状态；代码校验用户范围与本轮原话证据后持久化 |
| **主动补查与来源保留** | 初始历史不足时调用补查工具；新记忆保留有长度上限的用户原话，区分事件、咨询与系统保存时间 |
| **证据交付与复用** | Worker 返回实际检索摘录供总控核对；同一次调用内复用相同请求，并用引用代替重复文档正文 |
| **执行控制与可追溯状态** | 正式工具调用经过白名单、参数和作用域检查；配置调用上限、轮数与超时，记录实际执行及返回结果 |

[设计说明、代码入口与整体验证](results/agent-current-2026-09-09/README.md)。下图展开总控、专业 Agent 和实际注册的工具/Skills。

```mermaid
flowchart TB
    User["用户"] --> Supervisor["MedicalSupervisorAgent<br/>理解需求 · 按依赖调度"]
    Memory["患者档案 · 近期对话 · Mem0 历史"] --> Supervisor

    subgraph MemoryTools["总控记忆工具"]
        Profile["update_patient_profile<br/>更新患者档案"]
        History["search_patient_history<br/>补查跨会话历史"]
    end
    Supervisor --> Profile
    Supervisor --> History

    subgraph ConsultationGroup["健康咨询 Agent 与 Skills"]
        Consultation["Consultation Agent"]
        Lifestyle["recommend_lifestyle<br/>检索生活方式建议的候选资料"]
        Consultation --> Lifestyle
    end

    subgraph DiagnosticGroup["诊断 Agent 与 Skills"]
        Diagnostic["Diagnostic Agent"]
        Risk["assess_risk<br/>风险评估资料检索"]
        Symptoms["analyze_symptoms<br/>症状分析资料检索"]
        Code["disease_code<br/>ICD-10 编码查询"]
        Diagnostic --> Risk
        Diagnostic --> Symptoms
        Diagnostic --> Code
    end

    subgraph ResearchGroup["医学研究 Agent 与 Skills"]
        Research["Research Agent"]
        Guideline["clinical_guideline<br/>临床指南检索"]
        Knowledge["search_knowledge<br/>医学知识库检索"]
        DeepResearch["deep_research<br/>外部深度研究（需配置后端）"]
        Research --> Guideline
        Research --> Knowledge
        Research --> DeepResearch
    end

    Supervisor -->|call_consultation_agent| Consultation
    Supervisor -->|call_diagnostic_agent| Diagnostic
    Supervisor -->|call_research_agent| Research
    Consultation -.->|结果与实际证据| Summary["MedicalSupervisorAgent<br/>核对证据 · 汇总答复"]
    Diagnostic -.->|结果与实际证据| Summary
    Research -.->|结果与实际证据| Summary
    Supervisor -->|可直接回答| Summary
    Summary --> Answer["用户答复"]
```

图中列出当前注册的 **5 个总控工具和 7 个 Worker Skills**；节点名称对应实际调用函数。上下两个 `MedicalSupervisorAgent` 节点表示同一总控的调度与汇总阶段。总控记忆工具按用户身份和长期记忆配置启用，Worker 的 Skills 通过统一工具接口调用。[Agent 注册代码](medix-agent-swarm/agents/) · [Skills 实现](medix-agent-swarm/.claude/skills/)

## 快速开始

需要 Python 3.11+ 和可用的 OpenAI 兼容 API。以下为 PowerShell 命令：

```powershell
git clone https://github.com/qhzeng-gittec/medical-agent-assistant.git
cd medical-agent-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r medix-agent-swarm/requirements.txt -r requirements-dev.txt
Copy-Item config.example.py config.py
$env:LLM_API_KEY = "你的 API key"
$env:LLM_MODEL = "你的模型 ID"
$env:LLM_BASE_URL = "你的 OpenAI 兼容 API 地址"
python medix-agent-swarm/main.py
```

需要检索的问诊还需配置 Milvus、嵌入模型和知识库，见 [Agent 安装说明](medix-agent-swarm/README.md)。示例资料为合成技术文本。Linux/macOS 的环境设置与离线演示也见该文档。

## 关键测评

### Agent 整体测评

运行完整的 Supervisor、专业 Agent、患者档案、记忆和工具流程，检查系统能否完成多轮任务。**被测模型统一为 Qwen3.5-27B；MiniMax M2.5 与 Qwen3.5-27B 分别担任独立评审。**

| 能力 | 场景数 | 执行完成 | 有效双评 | 全部适用检查项通过 |
| --- | ---: | ---: | ---: | ---: |
| 多轮问诊与咨询 | 24 | 24 | 23 | **21/24** |
| 病史与跨会话记忆 | 24 | 24 | 24 | **23/24** |
| 证据检索与使用 | 24 | 24 | 24 | **23/24** |
| **合计** | **72** | **72** | **71** | **67/72（93.1%）** |

每类含常规、复杂、边界各 8 个合成场景；通过要求两名评审对全部适用项均判通过。1 场评分错误仍计入 72 场分母。该组是当前版本在已用于迭代的历史场景上的回归表现，逐轮核对回答、前后档案和工具证据；真实服务与受控证据的范围在报告中说明。

同批运行共记录 **365 次工具请求、195 次检索后端执行**，含预置历史的场景耗时中位数 **84.8 秒**。下方子任务并行对照来自另一组固定任务实验，按各自模型和计时范围解读。

[完整方法与结果](results/agent-current-2026-09-09/README.md) · [72 场逐项评分表](results/agent-current-2026-09-09/case_scores.csv) · [回答与双评理由](results/agent-current-2026-09-09/case_results.json) · [原始轨迹与评分](results/agent-current-2026-09-09/evidence.zip)

### Agent 调度效率

在固定的诊断、问诊和研究三个独立任务上，使用 **MiniMax M2.5 作为被测 Agent 模型**，串行与并行各执行 3 次。Worker 批次平均耗时由 **49.80 秒降至 20.15 秒，缩短 59.5%**。计时覆盖子任务执行阶段，总控规划和汇总另计。

两组使用同一模型、同一批任务，只改变调度方式；这是调度耗时对照。完整流程的请求数、费用和耗时另见 [架构测评](results/2026-09-08/architecture.md)。

### RAG 证据覆盖

在 **1,016 篇 MedlinePlus 健康主题摘要、800 条查询**的固定数据上，以 **Qwen3-Embedding-8B 生成检索向量**，按预先标注的目标来源直接计算覆盖，无需回答评分模型。测试集有 432 条完全可回答查询，保持同一排名，只改变返回的候选数量：

| 检索配置 | 完整目标来源覆盖 | 覆盖率 |
| --- | ---: | ---: |
| Top 1 | 333/432 | 77.1% |
| Top 3 | **408/432** | **94.4%** |
| Top 5 | 421/432 | 97.5% |

Top 3 比 Top 1 多覆盖 **75 个问题所需的全部来源**，说明保留多个候选能更好地支持多证据问题。统计使用无相似度过滤的固定排序，衡量检索覆盖；资料来自公开摘要，查询为合成检索任务。[数据、排名与复算](results/hybrid-grounding-2026-09-08/retrieval/summary.json)

### 工程回归测试

另有 **104 项 Agent 测试和 48 项评测运行器测试**通过，覆盖存储重开、用户隔离、工具协议、上下文预算和证据复用；网络输出使用确定性测试替身。工程测试与上方真实 API 场景测评分别计数。[测试代码](medix-agent-swarm/tests/)

### 模型训练与通用基准

模型训练使用 **Qwen3.5-2B**，产物为 CPT、LoRA SFT 和 GSPO 各阶段权重。这里评测训练后的模型；前面的 MiniMax 是 Agent 调度实验的被测模型，Qwen3-Embedding-8B 是检索编码器，三者承担不同任务。

使用原始 Qwen3.5-2B 与医学 CPT+SFT（`cpt_medical_v1`）在三个通用基准各 **500 道固定抽样题**上配对比较，共 **1,500 道独立题**：

| 基准与零样本候选似然指标 | 原始模型 | CPT+SFT | 变化（百分点） | 配对 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| MMLU 子集 · `acc` | 61.2% | 59.0% | −2.2 | [−5.6, +1.4] |
| ARC-Challenge · `acc_norm` | 44.2% | **51.2%** | **+7.0** | **[+3.8, +10.2]** |
| HellaSwag · `acc_norm` | 65.8% | 67.4% | +1.6 | [−0.6, +3.8] |

ARC-Challenge 的改善经三项主检验 Holm 校正后达到统计显著，另两项差值区间包含零。MMLU 排除六个医学科目；这是单训练种子和固定子集结果。[三项基准与离线复算](results/general-benchmarks-2026-09-08/README.md)

## 模型与数据

数据包含 **5,418 条训练、503 条验证、514 条测试记录及 314 张图片**，来自 VQA-RAD、MedMCQA、PubMedQA，含模型辅助标注。[数据来源与许可证](https://huggingface.co/datasets/starttoshow/medix-medical-sft)

已发布 A–F 六种 LoRA 配方、VQA GSPO 适配器，以及 [CPT→SFT→GSPO 四阶段权重链](https://huggingface.co/starttoshow/medix-qwen3.5-2b-cpt-sft-gspo-20260908)。权重链需按 CPT、SFT、RL 的依赖顺序合并，不能把后段适配器直接挂到原始基座；固定版本、哈希、配方和加载命令集中在 [训练与推理说明](MediX-R1/README.md)。

## 项目结构

```text
medical-agent-assistant/
├── medix-agent-swarm/    # Agent、技能、记忆、检索、示例与测试
├── MediX-R1/            # 训练、数据处理、评测与实验报告
├── results/             # 完整评测索引、汇总指标与复算工具
├── release/             # Hugging Face 产物上传工具
├── config.example.py    # API 与可选 Mem0 配置
└── requirements-dev.txt # 测试依赖
```

## 使用范围

项目用于研究和工程实验，尚未经过独立临床验证，不能替代专业诊疗。评测场景为合成资料或公开数据改编；上游来源与许可证随数据保留。运行时的密钥、患者档案和数据库应保存在各自的部署环境中。

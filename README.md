# Medical Agent Assistant

医疗多智能体助手与医疗模型训练实验。仓库包含两部分：由 Supervisor 调度的问诊 Agent 系统，以及 Qwen3.5-2B 的 LoRA 训练、GSPO 实验和评测代码。

Agent 当前调用配置的 OpenAI 兼容 API；训练模块独立运行。仓库没有将本地 LoRA 权重接入 Agent 的端到端效果验证。

## 项目结构

```text
medical-agent-assistant/
├── medix-agent-swarm/    # Supervisor、专业 Agent、Skills、记忆、检索及测试
├── MediX-R1/            # 模型训练、数据处理、GSPO、评测和实验报告
├── config.example.py    # Agent 配置模板；复制为 config.py 后使用
└── requirements-dev.txt # pytest
```

`MediX-R1` 是本仓库训练模块的目录名，原本地名称为 `medvlm_training_lab`。原参考框架目录未打包进入此仓库。

## Agent 系统

用户输入 → Patient Profile / Recent History / 可选 Mem0 → MedicalSupervisorAgent → Consultation、Diagnostic、Research 专业 Agent → 请求级 Evidence Store → 汇总回答。

- Supervisor 观察工具和专业 Agent 的返回结果后继续决策，支持有依赖的顺序执行与条件并行。
- Patient Profile 保存结构化事实；Recent History 按完整轮次管理预算；Mem0 提供可选跨会话记忆。
- Evidence Store 合并一次请求内相同参数的工具调用；RAG 上下文按来源复用正文。
- `.claude/skills/` 是运行时会加载的代码及技能描述，必须随源码保留。

安装、启动及开发评测见 [Agent 使用说明](medix-agent-swarm/README.md)。

## 训练与评测

[MediX-R1 使用说明](MediX-R1/README.md)列出数据准备、训练入口和测试前提。

- SFT：Attention 与 Attention+FFN LoRA，比较 VQA、知识、病例和上下文配方。
- GSPO：序列级比率、裁剪目标、rollout 和 VQA 强化学习实验；不把基础设施探针当作性能收益。
- 评测：分别评估答案与解释，保留数据划分、配对比较和局限性。

当前精选 [训练报告](MediX-R1/reports/training_report.md)包含 247 道项目留出题、944 份回答。该轮只评知识与上下文，不宣称四任务综合提升。

[Agent 开发测评报告](medix-agent-swarm/reports/医疗问诊Agent测评报告.md)记录 48 个主测任务；其中自动双评审只有 16/96 条有效评分，缺失评分未作为失败计入，也未据此发布整体医疗质量排名。报告是历史实验快照，原始运行轨迹未打包。

## 发布范围

仓库保留源码、技能、测试、开发场景、配置模板和精选报告。模型权重和主 SFT 数据集单独准备为 Hugging Face 发布包；代码仓库不存放这些大文件。虚拟环境、数据库、用户档案、原始 API 轨迹和录屏不发布。

评测目录里的 `private` 表示对被测模型隐藏的评分资料，不代表真实患者隐私数据。开发集包括合成场景及公开 CMB 数据改编，来源和上游许可证保留在数据集目录中。主 SFT 数据集包含处理后的 VQA-RAD、MedMCQA、PubMedQA；CPT 语料和本地医学指南文本暂未发布；知识库附带一条明确标注的技术演示资料。

实验报告中的路径已规范化，原始数据哈希保留；数据集文件清单的哈希对应本仓库公开版本。仅凭精选报告不能重算所有逐题成绩，完整复现实验仍需自行准备模型、数据和付费评测后端。

本项目用于学习和实验，尚未验证临床安全性，不能用于替代专业诊疗。

## 模型与数据发布

远端上传等待账号配置。已准备一份 SFT 数据集和 A–F 六组 SFT、GSPO 一组最终适配器，具体名称、基座版本和权重哈希见 [产物清单](MediX-R1/configs/artifacts.json)。所有权重均经过实际加载及 CPU 前向检查；数据导出后逐条还原核对，保持原文、划分、配方顺序与图片字节不变。

[Hugging Face 上传入口](release/publish_hf.py)默认只展示计划；配置本机 `hf auth login` 后，指定 `--packages`、`--namespace` 和 `--upload` 才会上传。不要把 token 放进源码或聊天。公开包约 502 MiB，未包含中途 checkpoint 和原始基座副本。

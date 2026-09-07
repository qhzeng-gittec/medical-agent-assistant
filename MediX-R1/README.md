# 医疗模型训练与评测

本目录来自项目自建的训练实验室，包含 Qwen3.5-2B 医疗多模态 SFT、GSPO 及评测代码。保持脚本与其相邻模块的原组织，便于追踪实验；它不包含原参考框架的整份副本。

## 主要入口

| 工作 | 代码 |
| --- | --- |
| VQA 项目划分与原图处理 | `prepare_vqa_project_split.py`、`original_images.py` |
| 推理监督与配方构建 | `complete_reasoning.py`、`build_native_recipes.py`、`prepare_knowledge_experiments.py` |
| LoRA SFT / CPT | `train_multitask_lora.py`、`train_cpt_lora.py` |
| GSPO rollout 与训练 | `train_vqa_gspo_probe.py` |
| 配方实验调度与恢复 | `run_knowledge_experiments.py` |
| 生成与评分 | `evaluate_native_reasoning.py`、`evaluate_vqa_rl.py`、`llm_judge.py` |
| 留出集及形式对照 | `run_independent_holdout.py`、`run_format_factorial.py` |
| 精选结果 | [训练报告](reports/training_report.md)、[指标](reports/report_metrics.json) |

## 环境与数据前提

训练环境与 Agent 环境分开安装。PowerShell 中进入本目录，运行 `./setup_env.ps1`，它会建立 `.venv` 并安装固定版本依赖及 CUDA PyTorch。CUDA 和 GPU 支持需要与实际机器匹配；`-SkipCudaTorch` 可跳过该脚本中的 PyTorch 安装。版本是原实验的环境记录，没有声明兼容所有机器。

代码默认从 `models/Qwen3.5-2B/` 加载本地模型。完整数据、图片、推理标注和生成后的配方应放在本目录 `data/` 下；它们与 `outputs/` 一同被 Git 忽略。

项目使用过 VQA-RAD、MedMCQA、PubMedQA 等公开来源，自行划分项目训练/验证/测试集。相关处理脚本保留来源标识和规则，但本仓库未提供完整数据的一键下载与重建，运行前须按脚本输入路径自行准备数据并遵守各上游条款。`medical_teacher.py` 调用已认证的 Codex CLI 完成标注和评分，相关步骤需要该外部工具、账号与对应模型访问权限。

准备好模型和 `data/native_reasoning/recipes/` 后，可以在本目录验证训练批次：

```powershell
.\.venv\Scripts\python.exe train_multitask_lora.py --recipe all_reasoning --dry-run
```

移除 `--dry-run` 才会训练。历史配方、留出集和诊断脚本依赖各自前序实验产物；缺少这些本地文件时不能直接重放历史实验。

## 测试

在训练环境中安装根目录 `requirements-dev.txt`，然后运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests --ignore=tests/test_native_reasoning.py -q
```

`test_native_reasoning.py` 是模型与数据集集成测试，需要本地模型及 `data/native_reasoning/` 后单独执行。其他测试覆盖序列级目标、评分解析、生成预算、并行标注和断点恢复。

## 结果边界

精选报告保存原始实验结论，使用仓库相对路径。`reports/holdout/` 保留方案、汇总成绩与隔离审计；完整逐题输入、生成回答、权重和日志不在仓库中，因此这份发布快照不足以独立重算全部历史成绩。

247 道留出题的报告只比较知识与上下文任务；[T0–T3 对照](reports/format_factorial.md)来自开发集，与留出测试分开。当前实验没有证明稳定临床收益，也没有证明训练后的模型改善了 Agent 端到端问诊表现。

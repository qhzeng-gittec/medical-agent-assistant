# 医疗模型训练与评测

Qwen3.5-2B 的医疗多模态 SFT、GSPO 及评测代码。训练模块与 Agent 系统独立运行。

## 目录与入口

| 目录 | 内容 | 主要入口 |
| --- | --- | --- |
| `training/` | LoRA 训练 | `python -m training.sft`、`python -m training.cpt`、`python -m training.gspo` |
| `data_processing/` | 下载发布集、清洗、划分、标注、配方构建 | `python -m data_processing.download_release` |
| `evaluation/` | 模型生成、LLM 评分、目标审计 | `python -m evaluation.evaluate_native_reasoning` |
| `experiments/` | 固定协议的配方对照、GSPO、消融和诊断 | `python -m experiments.run_knowledge_experiments`、`python -m experiments.run_gspo` |
| `common/` | 数据读写、图像处理、模板、标注后端 | 供各模块导入 |
| `configs/` | 基座版本、A–F 与 GSPO 权重清单 | `artifacts.json` |
| `reports/` | 精选成绩、方案及隔离审计 | [训练报告](reports/training_report.md) |
| `examples/` | 权重加载与 GSPO 目标示例 | `python -m examples.inference` |
| `tests/` | 离线回归与本地数据集成测试 | pytest |

所有模块命令在本目录执行。历史实验入口保留各自的数据和输出目录；共享函数已与一次性报告复制脚本分离。原始大文件和旧代码留在本地工作目录，没有直接删除实验原件。

## 环境

使用独立 Python 3.11+ 环境。在 PowerShell 中运行 `./setup_env.ps1`，它会建立 `.venv` 并安装原实验固定版本依赖及 CUDA PyTorch。CUDA 支持需与机器匹配；`-SkipCudaTorch` 跳过该脚本的 PyTorch 安装。运行下列命令前激活这个环境。

## 下载数据和模型

数据与七份最终适配器已准备为独立的 Hugging Face 发布包，远端 ID 在账号确认和上传后补齐。下面的 `HF_USERNAME` 与 `DATASET_COMMIT` 需要替换为实际账号和发布版本。

```powershell
python -m data_processing.download_release --repo-id HF_USERNAME/medix-medical-sft --revision DATASET_COMMIT
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir models/Qwen3.5-2B
```

下载器校验数据与图片哈希，恢复 `data/knowledge_experiments_v1/` 的三个 split、去重图片和四种配方顺序，拒绝覆盖已有数据目录。原始训练数据的本机图片路径转为可迁移路径；训练读取时相对 JSONL 文件解析。原始与发布版本哈希分别保留。

训练/验证/测试分别有 5,418 / 503 / 514 条；图片共 314 张。数据源为 VQA-RAD、MedMCQA、PubMedQA，包含模型辅助推理标注。CPT 指南语料和早期 pilot 的加工数据不在首版发布包中，对应实验需另外准备输入。

## 推理与训练

```powershell
python -m examples.inference --adapter HF_USERNAME/medix-qwen3.5-2b-knowledge-attention --question "What is the purpose of a randomized control group?"
python -m training.sft --recipes-dir data/knowledge_experiments_v1/recipes --recipe vqa_knowledge --lora-scope attention --dry-run
```

推理示例默认 CPU，使用 GPU 需显式加 `--device cuda`；VQA 加 `--image path/to/image.png`。移除训练命令的 `--dry-run` 才会训练。

完整六组对照由 `python -m experiments.run_knowledge_experiments` 调度。GSPO 入口为 `python -m experiments.run_gspo --output outputs/my_gspo_run`，需要先准备 C 组适配器于 `outputs/knowledge_experiments_v1/seed42/vqa_knowledge_attention/final_adapter/`。这两个流程使用 GPU，并通过已登录的 Codex CLI 调用外部模型评分，会消耗账号额度。

## 测试与结果

```powershell
python -m pip install -r ../requirements-dev.txt
python -m pytest tests --ignore=tests/test_native_reasoning.py -q
```

`test_native_reasoning.py` 需要额外的 `data/native_reasoning/` 和本地基座，不属于仅下载首版 SFT 发布集即可运行的离线测试。

[247 题留出测试报告](reports/training_report.md)只比较知识与上下文；[T0–T3 对照](reports/format_factorial.md)来自开发集。[GSPO 对比](reports/gspo_comparison.json)记录 93 道验证题、16 张图像，SFT 到 GSPO 的归一化评审分为 60.75 → 66.13；只有一个训练种子，不能表述为稳定诊断准确率提升。

数据包、配置和最终权重支持重新运行评测；精选报告未包含全部历史逐题评分文件，因此不能仅靠这些报告重算所有原始统计。尚未证明临床安全性或 Agent 端到端收益。

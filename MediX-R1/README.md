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

以下模块命令均在 `MediX-R1/` 目录执行。

## 环境

使用独立 Python 3.11+ 环境。在 PowerShell 中运行 `./setup_env.ps1`，它会建立 `.venv` 并安装原实验固定版本依赖及 CUDA PyTorch。CUDA 支持需与机器匹配；`-SkipCudaTorch` 跳过该脚本的 PyTorch 安装。运行下列命令前激活这个环境。

## 下载数据和模型

数据与七份最终适配器已公开发布至 [MediX 合集](https://huggingface.co/collections/starttoshow/medix-medical-sft-and-gspo-6a9e6f31b80642f4ba8b6f28)。以下下载命令固定数据版本；各适配器版本和权重哈希见 `configs/artifacts.json`。

```powershell
python -m data_processing.download_release --repo-id starttoshow/medix-medical-sft --revision 59b8da5629aac3f94ddaec9a9712b018daa81907
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir models/Qwen3.5-2B
```

下载器校验数据与图片哈希，恢复 `data/knowledge_experiments_v1/` 的三个 split、去重图片和四种配方顺序，拒绝覆盖已有数据目录。训练时相对 JSONL 所在目录解析图片路径。清单提供数据来源和文件哈希。

训练/验证/测试分别有 5,418 / 503 / 514 条；图片共 314 张。数据源为 VQA-RAD、MedMCQA、PubMedQA，包含模型辅助推理标注。CPT 指南语料和早期 pilot 的加工数据不在首版发布包中，对应实验需另外准备输入。

## 推理与训练

```powershell
python -m examples.inference --adapter starttoshow/medix-qwen3.5-2b-knowledge-attention --question "What is the purpose of a randomized control group?"
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

数据包、配置和最终权重支持重新运行评测。[完整结果](../results/README.md)提供逐题输出、评分和历史诊断附件，并附主要指标复算脚本。尚未证明临床安全性或 Agent 端到端收益。

## 新增 CPT→SFT→GSPO 实验链

[2026-09-08 模型包](https://huggingface.co/starttoshow/medix-qwen3.5-2b-cpt-sft-gspo-20260908)发布 CPT、依赖 CPT 的 SFT、直接 SFT 对照和依赖 CPT+SFT 的 RL 四组权重。固定版本与每个权重/config 哈希见 [expanded_chain.json](configs/expanded_chain.json)。按 BF16 顺序合并依赖，再挂载所选适配器；不要把依赖增量直接挂到原始基座。

从 `MediX-R1/` 执行：

```powershell
$env:HF_DEACTIVATE_ASYNC_LOAD = "1"
python -m examples.inference_chain --verify-only
python -m examples.inference_chain --stage rl --question "What is the purpose of a randomized control group?" --device cuda
python -m examples.inference_chain --stage sft_only --question "What is the purpose of a randomized control group?" --device cuda
```

默认下载固定 revision；`--package` 指向本地完整模型包，`--base-model` 指向本地 Qwen3.5-2B，默认设备为 CPU。演示是带总 token 上限的贪心生成，不能用两三个示例代替 512/384 token 机制评测。权重链不自动接入 Agent。

[阶段诊断](../results/2026-09-08/model_diagnostics.md)记录 150 知识题、91 个两问法事实族、312 知识/病例题以及 128 个 RL 训练题的冻结复评。部分知识参与过 CPT 选材；不是未见知识泛化测验。RL 未证明完整知识/病例净收益，原 A–F 与 VQA-GSPO 模型继续保留。

新增入口包括 `python -m training.train_cpt_rl_rounds`、`python -m experiments.run_cpt_coverage_experiment` 和 `python -m data_processing.prepare_cpt_expanded_candidate`。它们依赖已准备的 CPT 语料、训练数据、协议及上游适配器；完整 CPT 语料没有随此模型包发布。可选 Windows FLA/Triton 加速内核没有打包，默认使用普通 PyTorch 路径，只有自行准备对应环境后才设置 `MEDICAL_CPT_FAST_KERNELS=1`。

当前通用标注与 RL 评分入口使用已通过 ChatGPT 账号登录的 **Codex CLI / GPT-5.5**。`common/medical_teacher.py` 统一提交结构化输出请求，`common/codex_judge.py` 和 `common/codex_judge_transport.py` 保存请求、输出、调用事件及用量，`evaluation/rl_codex_judge.py` 为 RL 提供评分与调用预算管理。新评分缓存使用独立的模型命名空间。运行这些入口会使用已登录账号的额度；本次代码验证使用模拟调用，冻结成绩对应各报告中的原始运行。

历史 Gemini 评审对照入口 `compare_gemini_judge.py`、`optimize_gemini_judge.py`、`iterate_gemini_judge.py` 保留原来的模型、输出目录与汇总字段，对应原始报告。这些入口通过 `gemini_judge_bridge.mjs` 调用 Gemini，需要 Node.js 20+ 和 `GEMINI_API_KEY`，可用 `GEMINI_BASE_URL` 配置兼容地址；历史分数按原始模型标签保留。

新 600 题语义评分依赖此前选定的 round2 协议和本地评测输入，当前结果尚未完整评分。相关测试将真实数据/本地 tokenizer 检查与合成输入的单元测试分开，缺少前者时明确跳过。

发布前验证：Agent/评测工具 198 项离线测试通过；训练模块 64 项通过、3 项本地资料依赖检查跳过（另按既有说明排除 native 数据集成测试）。四组适配器哈希匹配；Windows CPU 上完成原始基座→合并 CPT→合并 SFT→挂载 RL 的真实加载和 2-token 生成冒烟。短生成只验证链路可执行，不是回答质量评分；未重新进行付费评审或全量 GPU 训练。

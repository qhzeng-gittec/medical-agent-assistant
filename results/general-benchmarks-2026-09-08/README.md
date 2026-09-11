# CPT+SFT 通用基准实测：MMLU、ARC-Challenge 与 HellaSwag

对原始 Qwen3.5-2B 与医学 CPT + Attention+FFN LoRA SFT 模型，使用 **MMLU、ARC-Challenge、HellaSwag 各 500 道固定抽样题**进行同题对比，共 **1,500 道独立题、两组模型**。三项零样本候选似然主指标完整列于下表，分别覆盖跨学科知识、科学问答与日常常识续写。

其中，ARC-Challenge 的 `acc_norm` 由 **44.2% 提升至 51.2%（+7.0 个百分点）**，配对 95% bootstrap 区间为 **[+3.8, +10.2]**。评测保留逐题分数、模型身份、冻结协议和离线复算入口。

这项实验检查：在医学文本上继续预训练（CPT），再用示范问答进行监督微调（SFT）后，模型在非医学任务上的表现如何变化。这里评的是模型对候选答案的打分，不是让医疗 Agent 调用工具回答问题。

表中的 `acc` 是选对答案的比例；`acc_norm` 先对候选答案分数作长度归一化，再计算选对比例，减少答案长度对排名的影响。两者都按标准答案自动计分。

## 三项基准完整主结果

| 基准与指标 | 原始 Qwen3.5-2B | 医学 CPT+SFT | 变化（百分点） | 配对 95% 区间 | Holm 校正 p |
| --- | ---: | ---: | ---: | --- | ---: |
| MMLU 子集 · `acc` | 306/500 · 61.2% | 295/500 · 59.0% | −2.2 | [−5.6, +1.4] | 0.400977 |
| ARC-Challenge · `acc_norm` | 221/500 · 44.2% | 256/500 · **51.2%** | **+7.0** | **[+3.8, +10.2]** | **0.000117** |
| HellaSwag · `acc_norm` | 329/500 · 65.8% | 337/500 · 67.4% | +1.6 | [−0.6, +3.8] | 0.400977 |

变化按“CPT+SFT − 原始模型”计算。ARC-Challenge 的提升在三项主检验经 Holm 校正后仍达到统计显著；MMLU 与 HellaSwag 的差值区间包含 0。配对变化与区间来自同一组题目；统计结果描述本次单训练种子、固定检查点的表现。三个任务分别计分，不合成为一个总分。

## 数据与评测方式

| 数据集 | 考察内容 | 本次使用的划分与抽样 |
| --- | --- | --- |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | 数学、历史、法律、经济、计算机等跨学科知识与解题 | test；排除六个医学科目，在其余 51 个学科内按学科比例抽取 500 题 |
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | 科学知识、因果关系与基础科学推理 | Challenge test；固定抽取 500 道科学考试选择题 |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | 日常活动常识与事件续写判断 | 有公开答案的 validation；固定抽取 500 题 |

抽样种子固定为 `20260908`，两组模型使用完全相同的题目和选项。三个基准的全部预设指标和辅助指标均在 [完整汇总表](comparison.csv) 与 [机器可读统计](results.json)中分别列出。

ARC 主评分采用 `Question: …\nAnswer:` 提示，计算各候选答案完整 token 序列的条件对数概率，再除以候选答案的字符数，选取最高分答案。它是答案选择指标，不给推理过程打分；无需调用外部模型裁判。MMLU 使用选项字母的候选似然，HellaSwag 使用续写候选的长度归一化似然。

所有主评测均为 zero-shot：使用原始选项顺序、不截断输入、每次计算一个无 padding 的候选，采用 BF16 权重。辅助聊天诊断使用同一批题、关闭 thinking、greedy 解码和 16-token 输出上限，记录首 token 选项排序、自由输出和格式遵循；该协议与主评分分别保存。

数据 revision、原始分片 SHA-256、抽样和训练重叠排查见 [数据清单](data/manifest.json)、[补充重叠核查](overlap_supplement.json)。[敏感性核查](overlap_sensitivity.json)另列排除一条通用短题干匹配后的计数。原模型预训练数据及语义改写重叠未被本次检查覆盖。

这是固定 500 题子集上的基准结果，不是完整榜单复现，也不是跨任务的能力不变性证明。完整 thinking 推理、中文、代码和视觉能力不属于本次测量范围。

## 对应的模型版本

基线是项目训练前的原始 Qwen3.5-2B。本次适配模型来自 **`cpt_medical_v1`，CPT rank 32 + Attention+FFN SFT rank 8**：先在 BF16 下合并 CPT，再加载 SFT 适配器。对应训练种子为 42。

| 权重 | SHA-256 |
| --- | --- |
| 原始基座权重分片 | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` |
| CPT 适配器 | `f8c3d6df1a89211d2a1af7d0c917237956d9422bc7db14d3827a21e6f91a64e0` |
| SFT 适配器 | `175132d744ad94ac8b6ffe21748519d6281212a9430f0631e73e46657351c17d` |

这是上述两份适配器的测评身份。仓库同日另一条 CPT→SFT→GSPO 发布链使用不同权重，按其自身报告解读。完整配置、版本与冻结文件哈希见 [plan.json](plan.json)。

## 数据、源码与复算

- [汇总 CSV](comparison.csv)：三个基准的主分数、配对区间、校正 p 值及辅助指标。
- [逐题对照 CSV](paired_predictions.csv)：1,500 道题的标签、选项、两模型选择与生成文本。
- [冻结题集](data/evaluation.jsonl)：题目、来源索引及实际评测提示所需字段。
- 原始逐题记录：[基线候选似然](runs/baseline/likelihood.jsonl)、[CPT+SFT 候选似然](runs/cpt_sft/likelihood.jsonl)、[基线聊天输出](runs/baseline/chat.jsonl)、[CPT+SFT 聊天输出](runs/cpt_sft/chat.jsonl)。只按 source ID 汇集为 JSONL，逐题内容保留。
- [评测源码快照](source/run_general_benchmarks.py)、[数据准备源码](source/prepare_general_benchmarks.py)、[协议调整记录](protocol_amendment.json)：保留正式运行前的计算路径检查依据；[原协议](plan_initial.json)与[预检记录](alignment_probe.json)可追溯。
- [发布清单](publication_manifest.json)：公开文件哈希及原始文件到公开副本的映射。机器路径转换为 `<EXPERIMENT_ROOT>`，源文件哈希与公开文件哈希分别记录。

从仓库根目录执行（Python 3.11+，无 GPU、无 API Key）：

```bash
python results/general-benchmarks-2026-09-08/reproduce.py
```

脚本校验公开文件哈希，从全部逐题记录复算三个基准的主指标、辅助指标、配对计数和精确 McNemar / Holm p 值。安装 NumPy 后可进一步逐项复算原始 bootstrap 区间：

```bash
python results/general-benchmarks-2026-09-08/reproduce.py --bootstrap
```

离线复算检查已保存输出的统计一致性，不重新生成模型答案。`source/` 是实验源码快照；重新推理需要对应哈希的模型、适配器、原实验目录布局及协议记录的依赖环境，权重不包含在本结果目录内。

## 来源与授权

题目与答案的来源、版本、数据卡见 [MMLU 数据卡](sources/mmlu_dataset_card.md)、[ARC 数据卡](sources/arc_challenge_dataset_card.md)、[HellaSwag 数据卡](sources/hellaswag_dataset_card.md)。原数据的权利归相应作者及来源方，数据卡保留原始署名；MMLU 使用 [MIT 许可](sources/mmlu_LICENSE)，ARC 标注 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，本目录 ARC 题目子集与格式改编按同一许可提供。HellaSwag 上游的 [MIT 许可](sources/hellaswag_LICENSE)一并保留，原作者仓库为 [rowanz/hellaswag](https://github.com/rowanz/hellaswag/)。本结果不将上游数据重新声明为独立原创数据。

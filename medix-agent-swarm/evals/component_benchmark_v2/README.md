# RAG 与 Mem0 独立组件测评

本套件将 RAG 的目标来源召回与 Mem0 的个人历史记忆分别评测，包含数据构建、冻结校验、真实 API 执行、逐项评分及结果复算。

直接下载：[1,016 篇语料](data/corpus.jsonl)、[400 条 RAG 查询](data/rag_cases.jsonl)、[120 个 Mem0 场景](data/mem0_cases.jsonl)、[数据冻结清单](data/freeze.json)、[标注复核](data/label_review.json)、[冻结前修订记录](data/curation_notes.json)。测评结果见 [RAG 报告](../../../results/component-benchmark-2026-09-08/rag/README.md)和 [Mem0 报告](../../../results/component-benchmark-2026-09-08/memory/README.md)。

## 数据与统计单位

| 套件 | 规模 | 划分与统计单位 |
| --- | --- | --- |
| RAG | 400 条查询，220 个场景族，MedlinePlus 健康主题摘要知识库 | 开发集 80 条、测试集 320 条；同源中英文改写及上下文配对在同一划分 |
| Mem0 | 120 个虚构用户场景，每场景 2 个会话、2 个正向历史查询 | 开发集 24 场景、测试集 96 场景；另外检查未知用户、其他应用和同主题无答案查询 |

RAG 的 400 条查询包括：240 条单来源中英文查询、80 条双来源查询、40 条上下文对照、40 条无答案查询。上下文对照的 20 条原始短句单列；剩余 340 条有明确目标来源的查询用于召回统计。同一场景的改写或重复不增加场景族数。

RAG 标签指向构造问题时采用的来源摘要，并保留其原文证据。指标为指定来源召回，库中其他材料也可能提供可接受答案。Mem0 对话为模型生成的虚构个人陈述，不含真实患者资料；按人物、时间、否定与不确定性核对记忆。

## 来源与授权

Source: MedlinePlus, National Library of Medicine.

知识库仅使用 [MedlinePlus 官方 XML](https://medlineplus.gov/xml.html)中的健康主题摘要。[使用说明](https://medlineplus.gov/about/using/usingcontent/)将这些摘要列为公共领域内容；本数据不包含第三方医学百科、药物专论、图片或标志。来源 URL、快照日期和文本处理规则随数据保存。

题目由 Qwen3.5-27B 辅助生成。源码中的固定随机种子先选择来源与划分；证据引文从明确编号的原文片段提取。模型辅助复核在执行检索前进行，冻结后通过 SHA-256 校验。该流程用于工程研究，临床应用需另行验证。

## 安装与运行

以下命令从仓库根目录执行，需要 Python 3.11+。`OPENROUTER_API_KEY` 从环境读取；执行与模型辅助评分会消耗 API 额度。

```powershell
python -m venv .venv-components
.\.venv-components\Scripts\Activate.ps1
python -m pip install -r medix-agent-swarm/evals/component_benchmark_v2/requirements.txt
python -m pytest medix-agent-swarm/evals/component_benchmark_v2/test_protocol.py -q
```

使用发布的冻结数据进行评测：

```powershell
python medix-agent-swarm/evals/component_benchmark_v2/evaluate_rag.py --output local-results/rag --cache local-cache/rag
python medix-agent-swarm/evals/component_benchmark_v2/evaluate_memory.py --split dev --output local-results/memory --stores local-cache/memory-stores
python medix-agent-swarm/evals/component_benchmark_v2/evaluate_memory.py --split test --output local-results/memory --stores local-cache/memory-stores
python medix-agent-swarm/evals/component_benchmark_v2/grade_memory.py --results local-results/memory --cache local-cache/memory-grades
```

RAG 使用真实 Qwen3-Embedding-8B API 与本地余弦排序。先在开发集选择阈值并写入 `selection.json`，再运行测试集；Top-K 对照另外在阈值 0 下统计。

Mem0 使用 `runtime/long_term.py` 中冻结的项目实现：Mem0 OSS 2.0.20、本地 Qdrant、Qwen3.5-27B 抽取和 Qwen3-Embedding-8B 向量。每个虚构用户有独立测试存储目录，写入两次会话后关闭并重新打开存储再查询；Top 3 / Top 10 均在固定阈值 0.3 下比较。语义覆盖由 MiniMax M2.5 按预设事实逐项判断，每项支持结论需给出可在返回记忆中找到的原文。词汇锚点和作用域隔离另计。

请求缓存与结果按固定参数校验；已完成的请求可复用。异常请求保留原始记录，再次执行前先检查；缺失项不计为完成。运行时的 Mem0 数据库保留在本地，RAG 的实际向量随结果发布。

离线复算已发布结果，无需 API Key：

```powershell
python medix-agent-swarm/evals/component_benchmark_v2/verify_rag_vectors.py --results results/component-benchmark-2026-09-08/rag
python medix-agent-swarm/evals/component_benchmark_v2/verify_results.py --results results/component-benchmark-2026-09-08
```

第一条命令需要 NumPy，从真实 API 向量重新计算 400 条查询的 Top 10 排名与余弦分数。第二条仅使用 Python 标准库，核对冻结数据、指标分母、逐项结果和语义评分引用。

## 重新构建数据

正式复算使用发布的冻结数据。以下脚本展示来源获取与构造过程；新一轮生成应使用单独版本目录，不能覆盖已冻结数据。

```powershell
python medix-agent-swarm/evals/component_benchmark_v2/fetch_sources.py --output local-sources
python medix-agent-swarm/evals/component_benchmark_v2/prepare.py --topics local-sources/topics.json --work local-generation
python medix-agent-swarm/evals/component_benchmark_v2/review_freeze.py --work local-generation
```

新生成结果先通过结构、原文引文、划分隔离与标签复核，再写入冻结清单。生成与复核模型版本、原始判定及人工修订如有发生均随数据记录。

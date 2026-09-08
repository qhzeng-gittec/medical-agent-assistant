# 混合检索与最终回答测评运行说明

本套件延续 component_benchmark_v2 的冻结语料、原有查询和实际向量；新增 BM25 / 加权 RRF、查询翻译和最终回答评测。混合索引实现在 [hybrid_retrieval.py](../../knowledge/hybrid_retrieval.py)。它是本轮本地快照检索适配器，产品默认 Milvus 路径没有被悄然替换。

## 离线复算

从仓库根目录运行，需要 Python 3.11+、NumPy 与 httpx（只用于导入在线运行器，验证不调用网络）：

```powershell
python -m pip install numpy httpx pytest
python -m pytest medix-agent-swarm/evals/hybrid_grounding_v3/test_hybrid.py -q
python medix-agent-swarm/evals/hybrid_grounding_v3/verify.py
```

验证冻结输入、场景隔离、全部 800 条排名、开发集参数选择和回答分母；评分原文引文与汇总一并核对。模型评审本身不等于离线客观真值，原始回答和评审理由可单独审查。

## 在线运行

设置自己的 `OPENROUTER_API_KEY`。下列命令调用实际 embedding、查询翻译、回答和评分 API，会消耗额度；不要将密钥写进代码或提交。

```powershell
python medix-agent-swarm/evals/hybrid_grounding_v3/retrieval.py
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage generate --domain rag
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage grade --domain rag
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage generate --domain memory
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage grade --domain memory
```

已完成结果按固定输入复用。要更换参数、模型或重新采样，应使用新的版本目录，不能覆盖这些冻结结果。数据构建使用 `prepare_codex.py --model gpt-5.5`，需要已登录 ChatGPT 的 Codex CLI；它只接收选定摘要生成 JSON，不访问检索结果。先前 API 作者的完成记录按逐源元数据保留。

RRF 网格与 tie-break 见检索协议；只有开发集决定参数。回答使用相同 Top-3 预算；普通提示对照仅用于新增的 160 条困难测试问题。Mem0 使用已保存的真实本地召回，不重新写入用户存储。

## 评分与历史记录

保留原始回答、提示、API 响应和逐项证据。格式错误只做有界修复，不因分数不理想重试。评分 v2 将不必要的拒答与肯定式事实编造分开；早期校准输出保留。记忆的 11 项标签修正见 `data/memory_label_corrections.json`，旧数据和旧分数不被覆盖。

`audit_judgments.py` 用 Codex CLI 独立复核全部初评失败和每种方法固定抽取的 8 份通过回答；结果是定向审计，不外推总体准确率，也不覆盖初评分数。CLI 遗漏项仅补该项；一处引文误字修正记录于 `judge_audit/quote_repairs.json`，保留原判定与原始输出。`answer_comparison.py` 生成配对区间与相同上下文敏感性分析，`case_studies.py` 导出全部失败索引，`costs.py` 汇总已有调用费用；`build_report.py` 从这些结果生成报告及哈希清单。

完整数据构成、数字和证据入口见[评测报告](../../../results/hybrid-grounding-2026-09-08/README.md)。

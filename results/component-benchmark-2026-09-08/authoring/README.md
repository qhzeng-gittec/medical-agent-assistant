# 数据构建与标注依据

正式评测使用 [冻结数据集](../../../medix-agent-swarm/evals/component_benchmark_v2/data/)。这里保留构建过程的 96 份真实 API 请求与响应：56 份 Qwen3.5-27B 生成记录和 40 份 MiniMax M2.5 标注复核记录，包含请求正文、模型响应、响应 ID 与用量，不含认证信息。

- [生成记录](generation/)：RAG 基于 MedlinePlus 健康主题摘要构造中英文检索问题；Mem0 构造虚构个人会话。
- [标注复核记录](label_review/)：核对来源证据、问题与预期事实的对应关系。
- [文件清单与原始下载哈希](manifest.json)：记录原始 XML 压缩包 SHA-256，以及各 API 记录的哈希。
- [最终复核结论](../../../medix-agent-swarm/evals/component_benchmark_v2/data/label_review.json)与[冻结前修订](../../../medix-agent-swarm/evals/component_benchmark_v2/data/curation_notes.json)：明确生成稿到冻结数据之间的整理。

复核记录中包含修订前后的标签检查；正式运行始终使用冻结后的同一版 JSONL。语料来源与授权、统计单位、开发/测试划分及复算命令见[套件说明](../../../medix-agent-swarm/evals/component_benchmark_v2/README.md)。

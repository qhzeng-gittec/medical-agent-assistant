---
name: deep-research
description: Search the live web with Tavily and return source text, URLs and retrieval time. Use for current medical information or source verification; inspect the returned evidence before drawing conclusions.
---

# Deep Research (深度研究)

通过 Tavily 实时搜索并返回网页正文或摘录，供调用 Agent 核验和综合。需要环境变量 `TAVILY_API_KEY`。

## When to Use

- 复杂的医学问题，需要多源信息综合
- 需要最新研究进展和文献综述
- 需要高置信度的证据支持

## 底层实现

- 数据源: Tavily Search API；知识库由其他工具独立查询
- 返回来源链接、可用的发布时间、检索时间和截断标记
- 不调用额外模型生成答案或证据评级；服务错误明确返回调用方

## 调用方式

```bash
/deep-research 糖尿病的最新治疗方法
```

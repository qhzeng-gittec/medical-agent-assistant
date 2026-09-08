---
name: assess-risk
description: Retrieve candidate medical evidence for risk assessment using the full case description, including negation, timing, subject and context. The agent determines urgency from the case and evidence; this tool does not classify risk.
---

# Assess Risk (风险评估)

检索风险评估候选资料，由诊断 Agent 结合完整病例判断风险和紧急程度。

## When to Use

- 用户描述症状，需要评估严重程度
- 判断是否需要紧急就医
- 风险分级（低/中/高/紧急）

## 底层实现

- 将模型提供的完整描述直接用于 Milvus 语义检索，不提取关键词、不匹配风险词表。
- 返回候选文档及 candidates / no_results 状态，不返回规则计算的风险等级。
- 保留否定、时间、主体与背景；无检索结果不代表低风险，检索失败显式上报。

## 调用方式

```bash
/assess-risk 胸痛,呼吸困难
```

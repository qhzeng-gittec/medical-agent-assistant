---
name: analyze-symptoms
description: Retrieve candidate evidence for risk assessment and symptom analysis from the full case description, preserving negation, timing, subject and context. The agent judges urgency, symptom relationships and differential diagnoses; this tool does not assign risk levels or map keywords to diseases.
---

# Analyze Symptoms (症状分析)

检索风险评估与症状分析候选资料，由诊断 Agent 判断紧急程度、症状关联和鉴别方向。

## When to Use

- 用户描述多个症状，需要模式分析
- 需要鉴别诊断建议
- 评估症状所涉及的身体系统

## 底层实现

- 将模型提供的完整描述直接用于 Milvus 语义检索，不拆分症状关键词。
- 不维护身体系统关键词表或系统到疾病的固定映射。
- 返回候选文档及 candidates / no_results 状态；相关性、风险和诊断思路由 Agent 结合上下文判断。

## 调用方式

```bash
/analyze-symptoms 头痛,发热,咳嗽
```

---
name: recommend-lifestyle
description: Retrieve candidate medical references for lifestyle questions. Results require relevance and applicability checks; they are not validated advice or personal patient history.
---

# Recommend Lifestyle (生活方式建议)

根据疾病或症状检索生活方式候选资料。由调用模型判断相关性和适用范围，资料不足时报告缺口。

## When to Use

- 用户问"高血压患者饮食注意什么""糖尿病如何运动"
- 需要生活方式调整建议
- 需要基础用药指导

## 底层实现

- 数据源: Milvus 向量数据库

## 调用方式

```bash
/recommend-lifestyle 高血压
```

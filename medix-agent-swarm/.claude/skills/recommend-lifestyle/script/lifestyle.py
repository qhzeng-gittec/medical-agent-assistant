"""
Recommend Lifestyle Skill
生活方式建议 Skill（自包含，无需依赖tools）
"""
from typing import Dict, Any
from loguru import logger
from core.rag_context import document_block

# 全局知识库实例
_kb_instance = None

def get_knowledge_base():
    global _kb_instance
    if _kb_instance is None:
        from knowledge.milvus_kb import MedicalKnowledgeBase
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


async def recommend_lifestyle(diagnosis: str) -> Dict[str, Any]:
    """
    检索生活方式候选资料，不评定其对当前患者的适用性

    Args:
        diagnosis: 疾病名称或症状

    Returns:
        {
            "answer": "格式化的生活方式建议",
            "diagnosis": "疾病名称",
            "categories": ["diet", "exercise", "lifestyle", "medication"]
        }
    """
    logger.info(f"Recommending lifestyle for: {diagnosis}")

    # 使用知识库单例
    kb = get_knowledge_base()

    # 从 Milvus 检索生活方式建议
    results = kb.search(
        query=diagnosis,
        top_k=3,
        filter_type="lifestyle"
    )

    if results:
        return {
            "answer": "以下为检索候选，不代表已匹配该疾病或适用于该患者。请依据资料实际主题和内容判断；无适用资料时明确说明证据不足。",
            "status": "candidates",
            "documents": [document_block(doc) for doc in results],
            "diagnosis": diagnosis,
            "categories": ["diet", "exercise", "lifestyle", "medication"],
            "source": "向量数据库"
        }
    else:
        # 未找到相关内容
        logger.warning(f"No lifestyle advice found in vector DB for {diagnosis}")
        return {
            "status": "no_results",
            "documents": [],
            "answer": f"未找到关于'{diagnosis}'的生活方式建议，请尝试更具体的疾病名称或联系医生咨询。",
            "diagnosis": diagnosis,
            "categories": [],
            "source": "未找到"
        }


def recommend_lifestyle_sync(diagnosis: str) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(recommend_lifestyle(diagnosis))

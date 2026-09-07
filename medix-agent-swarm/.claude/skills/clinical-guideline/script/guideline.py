"""
Clinical Guideline Skill
临床指南检索 Skill（自包含，无需依赖tools）
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


async def clinical_guideline(query: str, max_results: int = 1) -> Dict[str, Any]:
    """
    检索临床指南

    Args:
        query: 查询内容（疾病名称或治疗主题）
        max_results: 最大结果数（默认1，仅返回最相关的指南）

    Returns:
        {
            "answer": "格式化的临床指南信息",
            "guideline_title": "指南标题",
            "organization": "发布机构"
        }
    """
    logger.info(f"Searching clinical guidelines for: {query} (max_results={max_results})")

    # 使用知识库单例
    kb = get_knowledge_base()

    # 使用 Milvus 检索临床指南
    results = kb.search(
        query=f"{query} 临床指南 诊疗规范",
        top_k=max_results,  # 使用传入的 max_results 参数
        filter_type="clinical_guideline"
    )

    if results and results[0]["score"] > 0.1:
        doc = results[0]
        metadata = doc["metadata"]

        return {
            "answer": "临床指南正文见 documents，请核对来源、年份和适用范围。",
            "documents": [document_block(doc)],
            "guideline_title": f"{metadata.get('disease', query)}相关临床指南",
            "organization": metadata.get("organization", "N/A"),
            "year": metadata.get("year", "N/A"),
            "source": "向量数据库"
        }
    else:
        # 未找到相关内容
        logger.warning(f"No clinical guidelines found in vector DB for {query}")
        return {
            "answer": f"未找到'{query}'的相关临床指南，建议使用更具体的疾病名称或联系专业机构获取权威指南。",
            "guideline_title": "",
            "organization": "",
            "source": "未找到"
        }


def clinical_guideline_sync(query: str, max_results: int = 1) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(clinical_guideline(query, max_results))

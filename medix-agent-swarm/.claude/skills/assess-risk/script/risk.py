"""检索风险评估候选证据；风险判断由诊断 Agent 完成。"""
from typing import Any, Dict

from core.rag_context import document_block


_kb_instance = None


def get_knowledge_base():
    global _kb_instance
    if _kb_instance is None:
        from knowledge.milvus_kb import MedicalKnowledgeBase
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


async def assess_risk(symptoms: str) -> Dict[str, Any]:
    """用完整病例描述检索资料，保留否定、时间、主体和已知背景。"""
    if not isinstance(symptoms, str) or not symptoms.strip():
        raise ValueError("symptoms must be non-empty text")
    results = get_knowledge_base().search(query=symptoms, top_k=3, filter_type=None)
    return {
        "status": "candidates" if results else "no_results",
        "answer": "以下是候选资料，请结合完整病例判断风险与紧急程度；检索命中不代表适用，没有结果也不代表风险低或不存在疾病。",
        "documents": [document_block(doc) for doc in results],
    }


def assess_risk_sync(symptoms: str) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(assess_risk(symptoms))

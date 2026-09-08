"""
健康咨询Agent
支持 Skills 调用
"""
from typing import Dict, Any, Optional
import re

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin
from core import LLMClient


class ConsultationAgent(BaseAgent, SkillRegistryMixin):
    """
    健康咨询Agent
    通过 Skills 调用底层工具
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        default_config = {
            "model": "openai_compatible",
            "max_iterations": 5,
            "temperature": 0.8,
            "description": "健康咨询Agent，提供通用医疗咨询和健康建议"
        }

        config = config or default_config
        super().__init__(
            agent_id="consultation_agent",
            config=config,
            llm_client=llm_client
        )

        # 设置能力标签（Swarm 协作用）
        self.set_capabilities([
            "general_health_advice",
            "risk_assessment",
            "symptom_triage"
        ])

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是医疗健康咨询子 Agent，向主 Agent 交付所分派任务的结果。
围绕本次任务提供必要的解释或生活方式信息，不将资料整理自动扩展成建议。已有信息足够时直接完成；需要生活方式证据时可调用工具。工具返回的是候选资料，检查它的主题、适用人群和内容是否支持结论；没有适用证据就说明限制，通用知识与个人记录分开表达。
只补充分派任务所需的咨询分析，不重复总控已有内容。需要新的专项风险分析或超出可用工具的来源核验时，指出具体缺口，交由总控委派相应 Agent。仅引用实际获得的资料，不把自身知识称为检索结果，也不补造来源。
患者背景由主 Agent 提供，保留用户陈述和助手分析的区别；个人历史缺失时指出需要补查的内容。遵循已有风险评估，不作确诊、不开处方、不替代医生；紧急情况应及时就医。
简洁交付任务结论、必要来源和未解决的问题，格式按任务需要选择，面向用户的最终答复由主 Agent 组织。
工具调用通过工具接口执行，不把调用文本当作最终交付；工具失败或资料不足时，明确说明已完成和未完成的部分。
"""

    def register_tools(self):
        """只暴露健康咨询职责所需的 Skills。"""
        self.register_skills({"recommend_lifestyle"})

    async def post_process_result(
        self,
        result: Dict[str, Any],
        final_response: str
    ) -> Dict[str, Any]:
        """
        后处理：从最终响应中提取结构化信息
        """
        # 提取核心建议
        suggestions = []
        suggestion_pattern = r'【核心建议】\s*\n((?:\d+\.\s*.+\n?)+)'
        match = re.search(suggestion_pattern, final_response)

        if match:
            suggestion_text = match.group(1)
            suggestion_lines = re.findall(r'\d+\.\s*(.+)', suggestion_text)
            suggestions = [s.strip() for s in suggestion_lines if s.strip()]

        # 提取免责声明
        disclaimer_pattern = r'【免责声明】\s*\n(.+)'
        disclaimer_match = re.search(disclaimer_pattern, final_response)
        disclaimer = disclaimer_match.group(1) if disclaimer_match else \
            "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"

        result.update({
            'suggestions': suggestions[:5],  # 最多5条
            'disclaimer': disclaimer
        })

        return result


# 便捷函数
async def consult(question: str, **kwargs) -> Dict[str, Any]:
    """快捷咨询函数"""
    agent = ConsultationAgent()
    return await agent.process({'question': question, **kwargs})

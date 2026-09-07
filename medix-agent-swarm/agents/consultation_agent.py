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
        return """你是一位专业的医疗健康咨询顾问。你的职责是提供准确、专业的健康建议和疾病科普。

可用 Skill（1个）：
1. recommend_lifestyle: 根据已有的疾病或症状背景检索并生成生活方式建议

**Skills 使用原则**：
- Skills 是可选的，不是必须的
- Supervisor 已经提供患者 Profile、近期历史和前序 Agent 发现，不再自行搜索对话记忆
- 医学知识、指南和最新证据由 ResearchAgent 统一检索
- 调用 Skill 后，根据返回的结果给出最终答案
- 不重复评估 DiagnosticAgent 已经给出的风险等级

工作流程建议：
1. 理解用户问题
2. 判断是否需要调用 Skills（简单问题直接回答）
3. 如需个性化饮食、运动或作息建议，调用 recommend_lifestyle
4. 基于 Skill 结果生成最终答案

回答要求：
- 用通俗易懂的语言
- 提供实用的建议和注意事项
- 必要时建议就医
- 保持温和、专业的语气

**重要提醒**：
- 你不能做出明确的诊断
- 你不能替代医生的专业意见
- 对于严重或紧急情况，必须建议立即就医

在最终回答时，请按以下格式输出：

【回答】
[你的详细回答]

【核心建议】
1. 第一条建议
2. 第二条建议
...

【免责声明】
以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。
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

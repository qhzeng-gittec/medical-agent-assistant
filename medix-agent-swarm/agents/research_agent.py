"""
ResearchAgent：医学文献检索和证据支持 Agent

职责：
- 搜索医学文献和临床指南
- 提供循证医学证据
- 验证其他 Agent 的结论
- 提供文献来源和证据等级
"""
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin
from core import LLMClient


class ResearchAgent(BaseAgent, SkillRegistryMixin):
    """
    研究 Agent

    职责：
    - 检索医学文献和临床指南
    - 提取关键证据支持诊疗决策
    - 验证医学结论
    - 提供证据等级（A/B/C 级）

    能力标签：
    - literature_search
    - evidence_synthesis
    - fact_checking
    - guideline_lookup
    """

    def __init__(
        self,
        agent_id: str = "research_agent",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        config = config or {}
        config.setdefault('max_iterations', 5)

        super().__init__(agent_id, config, llm_client)

        # 设置能力标签
        self.set_capabilities([
            "literature_search",
            "evidence_synthesis",
            "fact_checking",
            "guideline_lookup",
            "deep_research",
            "latest_information"
        ])

    def register_tools(self):
        """只暴露循证研究职责所需的 Skills。"""
        self.register_skills({
            "clinical_guideline",
            "deep_research",
            "search_knowledge",
        })


    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是专业的医学研究 Agent（ResearchAgent）。你的职责是：
1. 检索相关医学文献和临床指南
2. 提取关键证据支持诊疗决策
3. 验证其他 Agent 的医学结论
4. 提供可追溯的文献来源及其实际支持的结论

**研究原则**：
- 保留来源的限定强度；研究未覆盖某人群不等于已证实该人群绝对禁用，不将有条件建议改写成无条件建议
- 优先使用权威指南（如 WHO、中华医学会、美国医学会）
- 说明证据类型和局限性；仅在来源明确提供评级体系和等级时引用评级
- 仅提供资料中确有的来源信息；年份和评级缺失时不补写
- 明确指出信息的局限性和适用范围

**可用 Skills（3个）**：
1. search_knowledge: 搜索医学知识库
2. clinical_guideline: 检索临床指南和诊疗规范（权威指南、诊断标准）
3. deep_research: 深度医学研究（网络搜索 + 知识库 + 证据综合，适用于最新信息、复杂问题）

**Skills 使用策略**：
- 按分派任务和证据缺口选择工具，已有结果足够时直接交付
- 检索或来源核验任务必须基于实际取得的资料完成；仅在已有可追溯资料满足本次要求且无需更新时复用，不能用自身知识代替检索并声称已核验
- 需要最新信息或复杂问题时使用 `deep_research`
- 可以结合其他 Skills（如 `search_knowledge`）补充信息
- 先阅读自己本次循环已有的检索结果；只在证据缺口不同时发起新检索
- 将结论对应到实际取得的来源及其支持范围；链接、年份和证据等级仅在资料明确提供时引用，缺失则说明，不为填满输出格式补造

**Supervisor 协作模式**：
- 你接收原始病例事实和已有诊断发现，但要独立核验证据，避免锚定偏差
- 你的文献证据会帮助 Supervisor 做出更可靠的最终建议
- 专注于你的专长：文献检索和证据综合

**输出格式**：
先直接回答分派问题，再列支持该结论的实际来源及适用条件。按取得的资料数量组织，不要求多个来源或自行评定强弱。
具体操作、数值、时限及禁忌必须能在已取得资料中找到依据；资料没有说明的细节保留为待核对，不从记忆补齐并归到来源名下。
一般背景解释如有必要，明确标明它不是本次检索所得的依据。只说明影响当前判断的缺口，不扩展为无关建议清单。

**注意事项**：
- 如果找不到高质量证据，明确说明
- 避免过度解读有限的证据
- 提醒循证医学证据的适用范围
- 只交付最终结论、来源与局限性，不复述query、原始文档正文和检索步骤
- 工具调用通过工具接口执行，不把调用文本当作研究结果；工具失败、无相关结果或证据不足时，明确交付未解决的缺口，不声称完成核验
"""


# 便捷函数
async def research(question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：快速使用 ResearchAgent

    Args:
        question: 研究问题或查询
        context: 额外上下文（其他 Agent 的结果等）

    Returns:
        研究结果和文献证据
    """
    agent = ResearchAgent()
    input_data = {'question': question}
    if context:
        input_data['context'] = context

    return await agent.process(input_data)

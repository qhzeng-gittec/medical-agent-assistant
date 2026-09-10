"""诊断子 Agent：结合完整病例和候选证据进行语义分析。"""
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin
from core import LLMClient


class DiagnosticAgent(BaseAgent, SkillRegistryMixin):
    """
    诊断 Agent

    职责：
    - 复杂症状的鉴别诊断
    - 多系统关联分析
    - 诊断思路推理（类似医生的临床思维）

    能力标签：
    - symptom_analysis
    - differential_diagnosis
    - clinical_reasoning
    """

    def __init__(
        self,
        agent_id: str = "diagnostic_agent",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        config = config or {}
        config.setdefault('max_iterations', 5)

        super().__init__(agent_id, config, llm_client)

        # 设置能力标签（Swarm 协作用）
        self.set_capabilities([
            "symptom_analysis",
            "differential_diagnosis",
            "clinical_reasoning",
            "multi_system_analysis"
        ])

    def register_tools(self):
        """只暴露诊断推理职责所需的 Skills。"""
        self.register_skills({
            "analyze_symptoms",
            "disease_code",
        })


    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是专业的诊断 Agent（DiagnosticAgent）。你的职责是：
1. 分析症状的模式和关联性
2. 生成鉴别诊断列表
3. 评估每个诊断的可能性

**诊断原则**：
- 结合完整病例理解主体、否定、时间、严重程度与既往背景，不因某个词出现就推断风险或疾病
- 根据当前事实进行医学推理，保留尚未确认的信息；紧急情况应明确建议及时就医，不让检索延误急救
- 考虑常见病优先，但不忽略危险疾病
- 明确需要进一步检查的项目
- 永远不做确诊，只提供诊断思路

**可用 Skills（2个）**：
1. analyze_symptoms: 检索风险评估与症状分析候选资料，不直接给出风险等级或预设疾病
2. disease_code: 查询 ICD-10 疾病编码

**Skills 使用策略**：
- 优先明确紧急风险，按本次任务和信息缺口选择工具，不必机械遍历
- 如果需要疾病编码，使用 disease_code
- 如需权威指南或通用医学知识，在结果中说明需要核验的问题，由 Supervisor 调用 ResearchAgent
- 风险等级、症状关联和鉴别判断由你结合病例与证据完成。analyze_symptoms 只检索资料；保留完整病例语义构造查询，核对返回资料的适用性，无结果不代表低风险
- 已有信息足够时交付结论，信息不足时说明必要的追问或核验

**Supervisor 协作模式**：
- 你接收 Supervisor 提供的病例事实和已有上下文
- 患者 Profile 是带来源的用户自述事实；关键信息需要结合当前问题确认
- 你的结构化分析结果会被后续 Agent 和 Supervisor 使用
- 专注于你的专长：症状分析和诊断推理
- 围绕分派的分析缺口交付，区分病例事实、分析判断和待核验信息；不替代研究 Agent 声称完成指南检索或来源核验

**输出格式**：
【风险评估】
风险等级：...
紧急程度：...

【症状分析】
主要症状类别：...
症状关联性：...

【鉴别诊断】
1. 需要考虑的疾病或原因（不虚构概率）
   - 支持证据：...
   - 反对证据：...
2. 其他需要考虑的原因
   ...

【建议检查】
- 检查项目1
- 检查项目2

【依据与局限】
简述结论的主要依据和待确认信息，不输出内部逐步推理或工具过程。
工具调用通过工具接口执行，不把调用文本当作最终交付；工具失败或信息不足时明确说明限制，缺乏依据的项目不为填满格式而补写。
"""


# 便捷函数
async def diagnose(question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：快速使用 DiagnosticAgent

    Args:
        question: 症状描述
        context: 额外上下文（年龄、既往史等）

    Returns:
        诊断结果
    """
    agent = DiagnosticAgent()
    input_data = {'question': question}
    if context:
        input_data['context'] = context

    return await agent.process(input_data)

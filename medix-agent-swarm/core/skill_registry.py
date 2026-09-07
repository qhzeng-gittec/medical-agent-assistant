"""
Skill 注册系统
直接将 Skill 函数转换为 OpenAI function calling 格式
"""
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import inspect
import asyncio
from loguru import logger


@dataclass
class SkillParameter:
    """Skill 参数定义"""
    name: str
    type: str  # "string", "number", "integer", "boolean", "object", "array"
    description: str
    required: bool = False
    enum: Optional[List[str]] = None


class SkillRegistry:
    """
    Skill 注册表

    存储 Skill 函数并提供执行和格式转换能力
    """

    def __init__(self, owner: str = "unknown_agent"):
        self.owner = owner
        self.skills: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: List[SkillParameter]
    ):
        """
        注册 Skill

        Args:
            name: Skill 名称
            function: Skill 函数（async 或 sync）
            description: Skill 描述
            parameters: 参数列表
        """
        self.skills[name] = {
            'function': function,
            'description': description,
            'parameters': parameters,
            'is_async': inspect.iscoroutinefunction(function)
        }
        logger.debug(f"Registered skill: {name}")

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """获取 Skill"""
        return self.skills.get(name)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 Skills"""
        return self.skills

    async def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        执行 Skill

        Args:
            name: Skill 名称
            **kwargs: Skill 参数

        Returns:
            Skill 执行结果
        """
        skill = self.skills.get(name)
        if not skill:
            error_msg = f"Skill not found: {name}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

        try:
            logger.debug(f"Executing skill: {name} with args: {kwargs}")

            async def invoke():
                if skill['is_async']:
                    return await skill['function'](**kwargs)
                return await asyncio.to_thread(skill['function'], **kwargs)

            from .evidence_store import EVIDENCE_SKILLS, current_evidence_store

            evidence_store = current_evidence_store()
            if evidence_store and name in EVIDENCE_SKILLS:
                result = await evidence_store.execute(
                    skill_name=name,
                    arguments=kwargs,
                    consumer=self.owner,
                    executor=invoke,
                )
            else:
                result = await invoke()

            logger.debug(f"Skill {name} completed successfully")
            return result

        except Exception as e:
            error_msg = f"Skill execution failed: {name} - {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "skill": name
            }

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        直接转换为 OpenAI function calling 格式

        Returns:
            OpenAI tools 格式的列表
        """
        tools = []

        for name, skill in self.skills.items():
            properties = {}
            required = []

            for param in skill['parameters']:
                prop = {
                    'type': param.type,
                    'description': param.description
                }
                if param.enum:
                    prop['enum'] = param.enum

                properties[param.name] = prop

                if param.required:
                    required.append(param.name)

            tools.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': skill['description'],
                    'parameters': {
                        'type': 'object',
                        'properties': properties,
                        'required': required
                    }
                }
            })

        return tools

"""
统一的 Skills 注册（自动发现）
所有 Worker Agents 共享
"""
from core.skill_loader import discover_skills
from core.skill_registry import SkillParameter
from pathlib import Path
from typing import Collection, Optional
from loguru import logger
import inspect


class SkillRegistryMixin:
    """
    统一注册所有 Skills（自动发现模式）

    所有 Worker Agents (ConsultationAgent, DiagnosticAgent, ResearchAgent)
    都继承这个 Mixin，避免重复代码
    """

    def register_skills(self, allowed_names: Optional[Collection[str]] = None):
        """
        自动扫描并注册所有 Skills

        Skills 会从 .claude/skills/ 目录自动发现，
        无需手动维护列表
        """
        project_root = Path(__file__).parent.parent
        discovered = discover_skills(project_root)

        allowed = set(allowed_names) if allowed_names is not None else None

        # 自动注册发现且属于当前 Agent 职责的 Skills
        for skill_info in discovered:
            function_name = skill_info["function_name"]
            if allowed is not None and function_name not in allowed:
                continue

            metadata = skill_info["metadata"]
            func = skill_info["function"]

            # 从 metadata 获取描述
            description = metadata.get("description", f"Skill: {skill_info['name']}")

            # 根据函数签名自动推断参数
            parameters = self._infer_skill_parameters(skill_info)

            # 注册到 SkillRegistry
            self.skill_registry.register(
                name=function_name,
                function=func,
                description=description,
                parameters=parameters
            )
            logger.info(f"✅ Registered skill: {function_name}")

        logger.info(f"Total {len(self.skill_registry.get_all())} skills registered")

    def register_all_skills(self):
        """注册全部 Skills；仅供确实需要通用工具集的 Agent 使用。"""
        self.register_skills()

    def _infer_skill_parameters(self, skill_info: dict) -> list:
        """
        从 skill 信息推断参数

        Args:
            skill_info: skill 信息字典

        Returns:
            [SkillParameter(...), ...]
        """
        func = skill_info["function"]

        # 获取函数签名
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            # 跳过 self 和特殊参数
            if param_name in ["self", "args", "kwargs"]:
                continue

            # 判断是否必需
            required = param.default == inspect.Parameter.empty

            # 推断类型（简单规则）
            param_type = "string"
            if "count" in param_name or "limit" in param_name or "max" in param_name or "iterations" in param_name:
                param_type = "number"

            # 生成描述
            param_desc = param_name.replace('_', ' ').title()

            parameters.append(SkillParameter(
                param_name,
                param_type,
                param_desc,
                required
            ))

        return parameters

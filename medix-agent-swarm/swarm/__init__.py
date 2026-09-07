"""医疗 Supervisor 与专业 Agent 协作入口。"""

from .supervisor_agent import (
    MedicalSupervisorAgent,
    process_medical_question,
)

__all__ = [
    'MedicalSupervisorAgent',
    'process_medical_question',
]

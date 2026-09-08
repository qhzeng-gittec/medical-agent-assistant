"""医疗 Supervisor 与专业 Agent 协作入口。"""

from .shared_context import SharedContext, SubTask, Contribution, TaskStatus
from .events import Event, EventType
from .supervisor_agent import (
    MedicalSupervisorAgent,
    process_medical_question,
)

__all__ = [
    'SharedContext',
    'SubTask',
    'Contribution',
    'TaskStatus',
    'Event',
    'EventType',
    'MedicalSupervisorAgent',
    'process_medical_question',
]

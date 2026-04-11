"""可插拔演化任务：每种任务提供种子、提示词、校验与评估。"""

from tasks.registry import TASK_REGISTRY, get_task, list_tasks

__all__ = ["TASK_REGISTRY", "get_task", "list_tasks"]

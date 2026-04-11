"""
兼容入口：提示词已迁至 ``tasks/min_max_layout_16/prompts.py``。
新任务请在任务包内实现 ``prompt_strategies``，由 ``EvolutionTask`` 暴露。
"""

from tasks.min_max_layout_16.prompts import (
    implement_exploration_prompt_strategy,
    implement_modification_prompt_strategy,
    implement_simplification_prompt_strategy,
)

__all__ = [
    "implement_modification_prompt_strategy",
    "implement_exploration_prompt_strategy",
    "implement_simplification_prompt_strategy",
]

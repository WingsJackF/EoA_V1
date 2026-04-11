import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: define_evaluator_api_interface

Description:
    Provides API interfaces for evaluating task code.
    Prefer the task-bound interface for multi-task runs.

Notes:
    - This module delegates actual evaluation to run_evaluation from the embedded
      evaluator and normalizes its output.
    - No file I/O, no modification of the evaluation logic.
"""

from typing import Dict, Any

# 默认仍指向当前内置的 16 点布局评估；多任务场景请使用 ``task.evaluate``。
from tasks.min_max_layout_16.evaluation import run_evaluation  # type: ignore
from tasks.base import EvolutionTask, normalize_standard_fitness


def evaluate_constructor_code(code_str: str) -> Dict[str, Any]:
    """
    Legacy API interface for evaluating a single-task code string.

    Args:
        code_str: The Python code string to evaluate.

    Returns:
        A standardized dictionary containing combined_score, eval_time, error,
        and any task-specific fields returned by the evaluator.
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    # Call the embedded evaluator logic (which handles its own exceptions
    # and returns a dict with evaluation details).
    result = run_evaluation(code_str)

    return normalize_standard_fitness(result)


def evaluate_task_code(task: EvolutionTask, code_str: str) -> Dict[str, Any]:
    """通用评估入口：交给任务对象自己的评估逻辑。"""
    if not isinstance(task, EvolutionTask):
        raise TypeError("task must be an EvolutionTask instance")
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")
    return task.evaluate(code_str)




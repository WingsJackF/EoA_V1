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
from tasks.base import EvolutionTask


def evaluate_constructor_code(code_str: str) -> Dict[str, Any]:
    """
    Legacy API interface for evaluating a single-task code string.

    Args:
        code_str: The Python code string to evaluate.

    Returns:
        A standardized dictionary containing min_max_ratio, combined_score,
        eval_time, and error (if any).
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    # Call the embedded evaluator logic (which handles its own exceptions
    # and returns a dict with evaluation details).
    result = run_evaluation(code_str)

    # Normalize/ensure keys exist in the returned dictionary.
    normalized: Dict[str, Any] = {
        "min_max_ratio": 0.0,
        "combined_score": 0.0,
        "eval_time": 0.0,
        "error": None,
    }

    if isinstance(result, dict):
        if "min_max_ratio" in result:
            try:
                normalized["min_max_ratio"] = float(result.get("min_max_ratio", 0.0))
            except (TypeError, ValueError):
                normalized["min_max_ratio"] = 0.0
        if "combined_score" in result:
            try:
                normalized["combined_score"] = float(result.get("combined_score", 0.0))
            except (TypeError, ValueError):
                normalized["combined_score"] = 0.0
        if "eval_time" in result:
            try:
                normalized["eval_time"] = float(result.get("eval_time", 0.0))
            except (TypeError, ValueError):
                normalized["eval_time"] = 0.0
        # Error handling: prefer explicit 'error' key, else try to find any error-like key
        if "error" in result and result.get("error") is not None:
            normalized["error"] = str(result.get("error"))
        else:
            # If result contains no error key but combined_score is zero and no min_max_ratio,
            # attempt to surface any message-like fields.
            if normalized["combined_score"] == 0.0 and normalized["min_max_ratio"] == 0.0:
                # Try common alternatives
                alt_error = result.get("message") or result.get("msg") or result.get("exception")
                if alt_error:
                    normalized["error"] = str(alt_error)
    else:
        # Unexpected non-dict result: provide a descriptive error
        normalized["error"] = "Evaluator returned non-dict result."

    return normalized


def evaluate_task_code(task: EvolutionTask, code_str: str) -> Dict[str, Any]:
    """通用评估入口：交给任务对象自己的评估逻辑。"""
    if not isinstance(task, EvolutionTask):
        raise TypeError("task must be an EvolutionTask instance")
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")
    return task.evaluate(code_str)




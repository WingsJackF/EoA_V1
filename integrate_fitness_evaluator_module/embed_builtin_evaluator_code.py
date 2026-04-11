"""
兼容入口：评估逻辑已迁至 ``tasks/min_max_layout_16/evaluation.py``。
新任务请实现各自的 ``EvolutionTask.evaluate_raw``，勿再依赖本模块。
"""

from tasks.min_max_layout_16.evaluation import run_evaluation

__all__ = ["run_evaluation"]

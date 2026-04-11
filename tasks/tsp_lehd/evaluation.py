"""任务专用评估：TSP LEHD。"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict

from tasks.task_support.gpu import apply_module_gpu_overrides, gpu_requested, logical_cuda_device_num
from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import (
    installed_module,
    load_program_module,
    local_problem_environment,
    resolve_callable,
)

POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")
TRAIN_PROBLEM_SIZE = 200
TRAIN_TEST_PARAS = {
    200: ["test_TSP200_n128.txt", 10, 10, 0],
    500: ["test_TSP500_n128.txt", 10, 10, 0],
    1000: ["test_TSP1000_n128.txt", 10, 10, 0],
}


def _prepare_candidate(program_code: str):
    candidate_module = load_program_module(program_code, module_name="tsp_lehd_candidate")
    heuristic = resolve_callable(candidate_module, POSSIBLE_NAMES)
    if not hasattr(candidate_module, "heuristics"):
        candidate_module.heuristics = heuristic
    if not hasattr(candidate_module, "heuristics_v2"):
        candidate_module.heuristics_v2 = heuristic
    return candidate_module


def run_evaluation(program_code: str) -> Dict[str, Any]:
    try:
        p = problem_dir("tsp_lehd")
        checkpoint = p / "checkpoints" / "checkpoint-150.pt"
        data_file = p / "data" / "test_TSP200_n128.txt"
        if not checkpoint.is_file():
            raise RuntimeError("需要预训练权重与数据，请将 checkpoint 放入 task_assets/problems/tsp_lehd/checkpoints")
        if not data_file.is_file():
            raise RuntimeError("需要测试数据，请将 LEHD 数据放入 task_assets/problems/tsp_lehd/data")
        candidate_module = _prepare_candidate(program_code)

        with local_problem_environment(p), installed_module("gpt", candidate_module):
            for module_name in ("eval", "utils", "TSPTester", "TSPModel", "TSPEnv"):
                sys.modules.pop(module_name, None)
            eval_mod = importlib.import_module("eval")
            apply_module_gpu_overrides(eval_mod)
            eval_mod.problem_size = TRAIN_PROBLEM_SIZE
            eval_mod.test_paras = dict(TRAIN_TEST_PARAS)
            raw = eval_mod.main_test(
                cuda_device_num=logical_cuda_device_num() if gpu_requested() else None
            )
        score_optimal, score_student, gap = raw
        score_optimal = float(score_optimal)
        score_student = float(score_student)
        gap = float(gap)
        return {
            "combined_score": -score_student,
            "eval_time": 0.0,
            "error": None,
            "teacher_score": score_optimal,
            "student_score": score_student,
            "gap_percent": gap,
        }
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}

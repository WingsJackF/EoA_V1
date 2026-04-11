"""任务专用评估：CVRP LEHD。"""

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
    200: ["vrp200_test_lkh.txt", 10, 10, 0],
    500: ["vrp500_test_lkh.txt", 10, 10, 0],
    1000: ["vrp1000_test_lkh.txt", 10, 10, 0],
}
FULL_TEST_PARAS = {
    "val": {
        200: ["vrp200_test_lkh.txt", 32, 32, 10],
        500: ["vrp500_test_lkh.txt", 32, 32, 10],
        1000: ["vrp1000_test_lkh.txt", 32, 32, 10],
    },
    "test": {
        200: ["vrp200_test_lkh.txt", 64, 64, 64],
        500: ["vrp500_test_lkh.txt", 64, 64, 64],
        1000: ["vrp1000_test_lkh.txt", 64, 64, 64],
    },
}
FULL_TEST_PROBLEM_SIZES = (200, 500, 1000)


def _prepare_candidate(program_code: str):
    candidate_module = load_program_module(program_code, module_name="cvrp_lehd_candidate")
    heuristic = resolve_callable(candidate_module, POSSIBLE_NAMES)
    if not hasattr(candidate_module, "heuristics"):
        candidate_module.heuristics = heuristic
    if not hasattr(candidate_module, "heuristics_v2"):
        candidate_module.heuristics_v2 = heuristic
    return candidate_module


def _ensure_assets():
    p = problem_dir("cvrp_lehd")
    checkpoint = p / "checkpoints" / "checkpoint-40.pt"
    if not checkpoint.is_file():
        raise RuntimeError("需要预训练权重与数据，请将 checkpoint 放入 task_assets/problems/cvrp_lehd/checkpoints")
    missing_files = [
        str((p / "data" / f"vrp{problem_size}_test_lkh.txt").relative_to(p))
        for problem_size in FULL_TEST_PROBLEM_SIZES
        if not (p / "data" / f"vrp{problem_size}_test_lkh.txt").is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "缺少 LEHD 测试数据文件："
            + ", ".join(missing_files)
            + "；请将数据放入 task_assets/problems/cvrp_lehd/data"
        )
    return p


def _evaluate_sizes(program_code: str, *, test_paras: Dict[int, list], problem_sizes: tuple[int, ...]) -> Dict[str, Dict[str, float]]:
    p = _ensure_assets()
    candidate_module = _prepare_candidate(program_code)
    results: Dict[str, Dict[str, float]] = {}
    with local_problem_environment(p), installed_module("gpt", candidate_module):
        for module_name in ("eval", "utils", "VRPTester", "VRPModel", "VRPEnv"):
            sys.modules.pop(module_name, None)
        eval_mod = importlib.import_module("eval")
        apply_module_gpu_overrides(eval_mod)
        eval_mod.test_paras = dict(test_paras)
        for problem_size in problem_sizes:
            eval_mod.problem_size = problem_size
            raw = eval_mod.main_test(
                cuda_device_num=logical_cuda_device_num() if gpu_requested() else None
            )
            score_optimal, score_student, gap = raw
            score_optimal = float(score_optimal)
            score_student = float(score_student)
            gap = float(gap)
            results[str(problem_size)] = {
                "teacher_score": score_optimal,
                "student_score": score_student,
                "gap_percent": gap,
                "combined_score": -score_student,
            }
    return results


def run_evaluation(program_code: str) -> Dict[str, Any]:
    try:
        per_size = _evaluate_sizes(
            program_code,
            test_paras=TRAIN_TEST_PARAS,
            problem_sizes=(TRAIN_PROBLEM_SIZE,),
        )
        metrics = per_size[str(TRAIN_PROBLEM_SIZE)]
        return {
            "combined_score": metrics["combined_score"],
            "eval_time": 0.0,
            "error": None,
            "teacher_score": metrics["teacher_score"],
            "student_score": metrics["student_score"],
            "gap_percent": metrics["gap_percent"],
        }
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        if mode not in FULL_TEST_PARAS:
            raise ValueError(f"Unsupported full test mode: {mode}")
        per_size = _evaluate_sizes(
            program_code,
            test_paras=FULL_TEST_PARAS[mode],
            problem_sizes=FULL_TEST_PROBLEM_SIZES,
        )
        mean_gap_percent = sum(item["gap_percent"] for item in per_size.values()) / len(per_size)
        return {
            "mode": mode,
            "problem_sizes": per_size,
            "mean_gap_percent": float(mean_gap_percent),
            "error": None,
        }
    except Exception as e:
        return {
            "mode": mode,
            "problem_sizes": {},
            "mean_gap_percent": None,
            "error": str(e),
        }

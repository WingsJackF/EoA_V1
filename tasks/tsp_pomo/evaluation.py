"""任务专用评估：TSP POMO。"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict

from tasks.task_support.gpu import apply_module_gpu_overrides
from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import (
    installed_module,
    load_program_module,
    local_problem_environment,
    resolve_callable,
)

POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")
TRAIN_PROBLEM_SIZE = 200
TRAIN_EPISODES = 10
TRAIN_BATCH_SIZE = 10
FULL_TEST_EPISODES = 64
FULL_TEST_BATCH_SIZE = 16


def _to_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _prepare_candidate(program_code: str):
    candidate_module = load_program_module(program_code, module_name="tsp_pomo_candidate")
    heuristic = resolve_callable(candidate_module, POSSIBLE_NAMES)
    if not hasattr(candidate_module, "heuristics"):
        candidate_module.heuristics = heuristic
    if not hasattr(candidate_module, "heuristics_v2"):
        candidate_module.heuristics_v2 = heuristic
    return candidate_module


def _evaluate_sizes(program_code: str, *, mode: str, problem_sizes: tuple[int, ...], episodes: int, batch_size: int) -> Dict[str, Dict[str, float]]:
    p = problem_dir("tsp_pomo")
    checkpoint = p / "checkpoints" / "checkpoint-3100.pt"
    if not checkpoint.is_file():
        raise RuntimeError("需要预训练权重与数据，请将 checkpoint 放入 task_assets/problems/tsp_pomo/checkpoints")
    candidate_module = _prepare_candidate(program_code)

    with local_problem_environment(p), installed_module("gpt", candidate_module):
        for module_name in ("eval", "utils", "gen_inst", "TSPTester", "TSPModel", "TSPEnv", "TSProblemDef"):
            sys.modules.pop(module_name, None)
        eval_mod = importlib.import_module("eval")
        apply_module_gpu_overrides(eval_mod)
        gen_inst_mod = importlib.import_module("gen_inst")
        dataset_conf = getattr(gen_inst_mod, "dataset_conf", {})
        if mode not in dataset_conf:
            raise ValueError(f"Unsupported full test mode: {mode}")
        results: Dict[str, Dict[str, float]] = {}
        for problem_size in problem_sizes:
            dataset_path = p / "dataset" / f"{mode}{problem_size}_dataset.pt"
            if not dataset_path.is_file():
                gen_inst_mod.generate_datasets(str(p / "dataset"))
            if not dataset_path.is_file():
                raise FileNotFoundError(f"缺少数据集 {dataset_path}")
            eval_mod.env_params["test_file_path"] = str(dataset_path)
            eval_mod.env_params["problem_size"] = problem_size
            eval_mod.tester_params["test_episodes"] = episodes
            eval_mod.tester_params["test_batch_size"] = batch_size
            objective = _to_float(eval_mod.main())
            results[str(problem_size)] = {
                "objective": objective,
                "combined_score": -objective,
            }
    return results


def run_evaluation(program_code: str) -> Dict[str, Any]:
    try:
        per_size = _evaluate_sizes(
            program_code,
            mode="train",
            problem_sizes=(TRAIN_PROBLEM_SIZE,),
            episodes=TRAIN_EPISODES,
            batch_size=TRAIN_BATCH_SIZE,
        )
        objective = per_size[str(TRAIN_PROBLEM_SIZE)]["objective"]
        return {
            "combined_score": -objective,
            "eval_time": 0.0,
            "error": None,
            "objective": objective,
        }
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        p = problem_dir("tsp_pomo")
        with local_problem_environment(p):
            gen_inst_mod = importlib.import_module("gen_inst")
            dataset_conf = getattr(gen_inst_mod, "dataset_conf", {})
        if mode not in dataset_conf:
            raise ValueError(f"Unsupported full test mode: {mode}")
        per_size = _evaluate_sizes(
            program_code,
            mode=mode,
            problem_sizes=tuple(dataset_conf[mode]),
            episodes=FULL_TEST_EPISODES,
            batch_size=FULL_TEST_BATCH_SIZE,
        )
        mean_combined_score = float(sum(item["combined_score"] for item in per_size.values()) / len(per_size))
        return {
            "mode": mode,
            "problem_sizes": per_size,
            "mean_combined_score": mean_combined_score,
            "error": None,
        }
    except Exception as e:
        return {"mode": mode, "problem_sizes": {}, "mean_combined_score": None, "error": str(e)}

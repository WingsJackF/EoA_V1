"""任务专用评估：Decap placement GA crossover。"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import import_problem_module, installed_module, load_program_module, resolve_callable

POSSIBLE_NAMES = ("crossover", "crossover_v1", "crossover_v2", "crossover_v3")


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"min_max_ratio": 0.0, "combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"min_max_ratio": 0.0, "combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        candidate_module = load_program_module(program_code, module_name="dpp_ga_candidate")
        crossover = resolve_callable(candidate_module, POSSIBLE_NAMES)
        prob = problem_dir("dpp_ga")

        with installed_module("gpt", candidate_module):
            eval_mod = import_problem_module(prob, "eval")
        reward_mod = import_problem_module(prob, "reward_functions")

        reward_model = reward_mod.RewardModel(str(prob), n=10, m=10, model_number=5, freq_pts=201)
        eval_mod.crossover = crossover
        eval_mod.test_probe = np.load(prob / "test_problems" / "test_100_probe.npy")[:2]
        eval_mod.test_prohibit = np.load(prob / "test_problems" / "test_100_keepout.npy")[:2]
        eval_mod.keepout_num = np.load(prob / "test_problems" / "test_100_keepout_num.npy")[:2]
        eval_mod.n = 10
        eval_mod.m = 10

        return float(eval_mod.run_ga(16, 3, 2, 0.2, 20, reward_model))

    return _wrap(inner)

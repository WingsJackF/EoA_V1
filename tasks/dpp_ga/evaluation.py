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
        return {"combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


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


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        if mode not in {"val", "test"}:
            raise ValueError(f"Unsupported full test mode: {mode}")
        candidate_module = load_program_module(program_code, module_name="dpp_ga_full_candidate")
        crossover = resolve_callable(candidate_module, POSSIBLE_NAMES)
        prob = problem_dir("dpp_ga")

        with installed_module("gpt", candidate_module):
            eval_mod = import_problem_module(prob, "eval")
        reward_mod = import_problem_module(prob, "reward_functions")

        reward_model = reward_mod.RewardModel(str(prob), n=10, m=10, model_number=5, freq_pts=201)
        eval_mod.crossover = crossover
        test_probe = np.load(prob / "test_problems" / "test_100_probe.npy")
        test_prohibit = np.load(prob / "test_problems" / "test_100_keepout.npy")
        keepout_num = np.load(prob / "test_problems" / "test_100_keepout_num.npy")
        eval_mod.n = 10
        eval_mod.m = 10
        if mode == "val":
            eval_mod.test_probe = test_probe[5:10]
            eval_mod.test_prohibit = test_prohibit[5:10]
            eval_mod.keepout_num = keepout_num[5:10]
            avg_reward = float(eval_mod.run_ga(20, 20, 5, 0.2, 20, reward_model))
        else:
            eval_mod.test_probe = test_probe[-64:]
            eval_mod.test_prohibit = test_prohibit[-64:]
            eval_mod.keepout_num = keepout_num[-64:]
            avg_reward = float(eval_mod.run_ga(20, 10, 64, 0.2, 20, reward_model))
        return {
            "mode": mode,
            "problem_sizes": {
                mode: {
                    "average_reward": avg_reward,
                    "combined_score": avg_reward,
                }
            },
            "mean_combined_score": avg_reward,
            "error": None,
        }
    except Exception as e:
        return {"mode": mode, "problem_sizes": {}, "mean_combined_score": None, "error": str(e)}

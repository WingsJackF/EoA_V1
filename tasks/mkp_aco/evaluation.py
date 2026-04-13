"""任务专用评估：MKP ACO。"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch

from tasks.task_support.gpu import solver_device
from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import import_problem_module, load_program_module, resolve_callable

MAX_INSTANCES = 5
N_ITERATIONS = 50
N_ANTS = 10
POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")
FULL_TEST_SIZES = {
    "val": (100, 300, 500),
    "test": (100, 200, 300, 500, 1000),
}


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="mkp_aco_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("mkp_aco")
        aco_mod = import_problem_module(base, "aco")
        ds = base / "dataset" / "train100_dataset.npz"
        if not ds.is_file():
            ds = base / "dataset" / "train50_dataset.npz"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {base / 'dataset'} 下的 train100/train50 数据")
        data = np.load(ds)
        prizes, weights = data["prizes"], data["weights"]
        k = min(MAX_INSTANCES, prizes.shape[0])
        objs = []
        for i in range(k):
            heu = heuristics(prizes[i].copy(), weights[i].copy()) + 1e-9
            if tuple(heu.shape) != (weights[i].shape[0],):
                raise ValueError(f"Invalid heuristic shape: {heu.shape}")
            heu[heu < 1e-9] = 1e-9
            solver = aco_mod.ACO(
                torch.from_numpy(prizes[i]),
                torch.from_numpy(weights[i]),
                torch.from_numpy(heu),
                N_ANTS,
                device=solver_device(),
            )
            obj, _ = solver.run(N_ITERATIONS)
            objs.append(float(obj.item() if hasattr(obj, "item") else obj))
        return float(np.mean(objs))

    return _wrap(inner)


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        if mode not in FULL_TEST_SIZES:
            raise ValueError(f"Unsupported full test mode: {mode}")
        module = load_program_module(program_code, module_name="mkp_aco_full_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("mkp_aco")
        aco_mod = import_problem_module(base, "aco")
        train_ds = base / "dataset" / "train100_dataset.npz"
        if not train_ds.is_file():
            gen_inst_mod = import_problem_module(base, "gen_inst")
            gen_inst_mod.generate_datasets()
        size_to_objective: Dict[str, Dict[str, float]] = {}
        for problem_size in FULL_TEST_SIZES[mode]:
            ds = base / "dataset" / f"{mode}{problem_size}_dataset.npz"
            if not ds.is_file():
                raise FileNotFoundError(f"缺少数据集 {ds}")
            data = np.load(ds)
            prizes, weights = data["prizes"], data["weights"]
            objs = []
            for i in range(prizes.shape[0]):
                heu = heuristics(prizes[i].copy(), weights[i].copy()) + 1e-9
                if tuple(heu.shape) != (weights[i].shape[0],):
                    raise ValueError(f"Invalid heuristic shape: {heu.shape}")
                heu[heu < 1e-9] = 1e-9
                solver = aco_mod.ACO(torch.from_numpy(prizes[i]), torch.from_numpy(weights[i]), torch.from_numpy(heu), N_ANTS, device=solver_device())
                obj, _ = solver.run(N_ITERATIONS)
                objs.append(float(obj.item() if hasattr(obj, "item") else obj))
            avg_obj = float(np.mean(objs))
            size_to_objective[str(problem_size)] = {
                "objective": avg_obj,
                "combined_score": avg_obj,
            }
        mean_combined_score = float(np.mean([x["combined_score"] for x in size_to_objective.values()]))
        return {"mode": mode, "problem_sizes": size_to_objective, "mean_combined_score": mean_combined_score, "error": None}
    except Exception as e:
        return {"mode": mode, "problem_sizes": {}, "mean_combined_score": None, "error": str(e)}

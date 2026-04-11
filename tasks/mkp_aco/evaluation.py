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
        return -float(np.mean(objs))

    return _wrap(inner)

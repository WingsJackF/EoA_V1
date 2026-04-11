"""任务专用评估：TSP ACO。"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from scipy.spatial import distance_matrix

from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import import_problem_module, load_program_module, resolve_callable

MAX_INSTANCES = 5
N_ITERATIONS = 100
N_ANTS = 30
POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"min_max_ratio": 0.0, "combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"min_max_ratio": 0.0, "combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="tsp_aco_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("tsp_aco")
        aco_mod = import_problem_module(base, "aco")
        ds = base / "dataset" / "train50_dataset.npy"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {ds}，请在 task_assets/problems/tsp_aco 下运行 gen_inst.generate_datasets()")
        nodes = np.load(ds)
        k = min(MAX_INSTANCES, len(nodes))
        objs = []
        for i in range(k):
            dist_mat = distance_matrix(nodes[i], nodes[i])
            dist_mat[np.diag_indices_from(dist_mat)] = 1
            heu = heuristics(dist_mat.copy()) + 1e-9
            heu[heu < 1e-9] = 1e-9
            solver = aco_mod.ACO(dist_mat, heu, n_ants=N_ANTS)
            obj = solver.run(N_ITERATIONS)
            objs.append(float(obj.item() if hasattr(obj, "item") else obj))
        return -float(np.mean(objs))

    return _wrap(inner)

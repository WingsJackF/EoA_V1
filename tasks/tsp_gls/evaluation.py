"""任务专用评估：TSP GLS。"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import import_problem_module, load_program_module, resolve_callable

MAX_INSTANCES = 5
POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")
PERTURBATION_MOVES = 30
ITER_LIMIT = 1200


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="tsp_gls_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("tsp_gls")
        gls_mod = import_problem_module(base, "gls")
        gen_inst_mod = import_problem_module(base, "gen_inst")
        ds = base / "dataset" / "train200_dataset.npy"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {ds}，请在 task_assets/problems/tsp_gls 下运行 gen_inst.generate_datasets()")
        instances = gen_inst_mod.load_dataset(str(ds))
        k = min(MAX_INSTANCES, len(instances))
        objs = []
        for i in range(k):
            inst = instances[i]
            heu = heuristics(inst.distmat.copy())
            if tuple(heu.shape) != (inst.n, inst.n):
                raise ValueError(f"Invalid heuristic shape: {heu.shape}")
            result = gls_mod.guided_local_search(inst.distmat, heu, PERTURBATION_MOVES, ITER_LIMIT)
            objs.append(float(inst.distmat[result, np.roll(result, 1)].sum().item()))
        return -float(np.mean(objs))

    return _wrap(inner)

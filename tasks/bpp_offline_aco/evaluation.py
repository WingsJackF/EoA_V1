"""任务专用评估：Offline BPP ACO。"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import import_problem_module, load_program_module, resolve_callable

MAX_INSTANCES = 5
N_ANTS = 20
SAMPLE_COUNT = 200
POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"min_max_ratio": 0.0, "combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"min_max_ratio": 0.0, "combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="bpp_offline_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("bpp_offline_aco")
        aco_mod = import_problem_module(base, "aco")
        gen_inst_mod = import_problem_module(base, "gen_inst")
        ds = base / "dataset" / "train500_dataset.npz"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {ds}，请在 task_assets/problems/bpp_offline_aco 下运行 gen_inst.generate_datasets()")
        dataset = gen_inst_mod.load_dataset(str(ds))
        k = min(MAX_INSTANCES, len(dataset))
        objs = []
        for i in range(k):
            inst = dataset[i]
            heu = heuristics(inst.demands.copy(), inst.capacity)
            if tuple(heu.shape) != (inst.n, inst.n):
                raise ValueError(f"Invalid heuristic shape: {heu.shape}")
            if not (0 < heu.max() < np.inf):
                raise ValueError("Heuristic matrix must contain positive finite values")
            solver = aco_mod.ACO(inst.demands, heu.astype(float), capacity=inst.capacity, n_ants=N_ANTS, greedy=False)
            obj, _ = solver.sample_only(SAMPLE_COUNT)
            objs.append(float(obj.item() if hasattr(obj, "item") else obj))
        return -float(np.mean(objs))

    return _wrap(inner)

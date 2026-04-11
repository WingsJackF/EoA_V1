"""任务专用评估：OP ACO。"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch

from tasks.task_support.gpu import solver_device
from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import import_problem_module, load_program_module, resolve_callable

MAX_INSTANCES = 5
N_ITERATIONS = 50
N_ANTS = 20
POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="op_aco_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("op_aco")
        aco_mod = import_problem_module(base, "aco")
        gen_inst_mod = import_problem_module(base, "gen_inst")
        ds = base / "dataset" / "train50_dataset.npz"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {ds}，请在 task_assets/problems/op_aco 下运行 gen_inst.generate_datasets()")
        dataset = gen_inst_mod.load_dataset(str(ds))
        k = min(MAX_INSTANCES, len(dataset))
        objs = []
        for i in range(k):
            inst = dataset[i]
            heu = heuristics(np.array(inst.prize), np.array(inst.distance), inst.maxlen) + 1e-9
            if tuple(heu.shape) != (inst.n, inst.n):
                raise ValueError(f"Invalid heuristic shape: {heu.shape}")
            heu[heu < 1e-9] = 1e-9
            solver = aco_mod.ACO(
                inst.prize,
                inst.distance,
                inst.maxlen,
                torch.from_numpy(heu),
                N_ANTS,
                device=solver_device(),
            )
            obj, _ = solver.run(N_ITERATIONS)
            objs.append(float(obj.item() if hasattr(obj, "item") else obj))
        return -float(np.mean(objs))

    return _wrap(inner)

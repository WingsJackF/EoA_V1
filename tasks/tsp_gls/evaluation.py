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


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        module = load_program_module(program_code, module_name="tsp_gls_full_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("tsp_gls")
        gls_mod = import_problem_module(base, "gls")
        gen_inst_mod = import_problem_module(base, "gen_inst")
        dataset_conf = getattr(gen_inst_mod, "dataset_conf", {})
        if mode not in dataset_conf:
            raise ValueError(f"Unsupported full test mode: {mode}")
        required_train = base / f"dataset/train{dataset_conf['train'][0]}_dataset.npy"
        if not required_train.is_file():
            gen_inst_mod.generate_datasets()
        size_to_objective: Dict[str, Dict[str, float]] = {}
        for problem_size in dataset_conf[mode]:
            ds = base / "dataset" / f"{mode}{problem_size}_dataset.npy"
            if not ds.is_file():
                raise FileNotFoundError(f"缺少数据集 {ds}")
            instances = gen_inst_mod.load_dataset(str(ds))
            objs = []
            for inst in instances:
                heu = heuristics(inst.distmat.copy())
                if tuple(heu.shape) != (inst.n, inst.n):
                    raise ValueError(f"Invalid heuristic shape: {heu.shape}")
                result = gls_mod.guided_local_search(inst.distmat, heu, PERTURBATION_MOVES, ITER_LIMIT)
                objs.append(float(inst.distmat[result, np.roll(result, 1)].sum().item()))
            avg_obj = float(np.mean(objs))
            size_to_objective[str(problem_size)] = {
                "objective": avg_obj,
                "combined_score": -avg_obj,
            }
        mean_combined_score = float(np.mean([x["combined_score"] for x in size_to_objective.values()]))
        return {"mode": mode, "problem_sizes": size_to_objective, "mean_combined_score": mean_combined_score, "error": None}
    except Exception as e:
        return {"mode": mode, "problem_sizes": {}, "mean_combined_score": None, "error": str(e)}

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
N_ITERATIONS = 15


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


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


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        module = load_program_module(program_code, module_name="bpp_offline_full_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("bpp_offline_aco")
        aco_mod = import_problem_module(base, "aco")
        gen_inst_mod = import_problem_module(base, "gen_inst")
        dataset_conf = getattr(gen_inst_mod, "dataset_conf", {})
        if mode not in dataset_conf:
            raise ValueError(f"Unsupported full test mode: {mode}")
        required_train = base / f"dataset/train{dataset_conf['train'][0]}_dataset.npz"
        if not required_train.is_file():
            gen_inst_mod.generate_datasets()
        size_to_objective: Dict[str, Dict[str, float]] = {}
        for problem_size in dataset_conf[mode]:
            ds = base / "dataset" / f"{mode}{problem_size}_dataset.npz"
            if not ds.is_file():
                raise FileNotFoundError(f"缺少数据集 {ds}")
            dataset = gen_inst_mod.load_dataset(str(ds))
            objs = []
            for inst in dataset:
                heu = heuristics(inst.demands.copy(), inst.capacity)
                if tuple(heu.shape) != (inst.n, inst.n):
                    raise ValueError(f"Invalid heuristic shape: {heu.shape}")
                if not (0 < heu.max() < np.inf):
                    raise ValueError("Heuristic matrix must contain positive finite values")
                solver = aco_mod.ACO(inst.demands, heu.astype(float), capacity=inst.capacity, n_ants=N_ANTS, greedy=False)
                obj, _ = solver.run(N_ITERATIONS)
                objs.append(float(obj.item() if hasattr(obj, "item") else obj))
            avg_obj = float(np.mean(objs))
            size_to_objective[str(problem_size)] = {
                "objective": avg_obj,
                "combined_score": -avg_obj,
            }
        mean_combined_score = float(np.mean([x["combined_score"] for x in size_to_objective.values()]))
        return {"mode": mode, "problem_sizes": size_to_objective, "mean_combined_score": mean_combined_score, "error": None}
    except Exception as e:
        return {"mode": mode, "problem_sizes": {}, "mean_combined_score": None, "error": str(e)}

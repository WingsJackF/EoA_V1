"""任务专用评估：TSP constructive。"""

from __future__ import annotations

from copy import copy
from typing import Any, Callable, Dict

import numpy as np
from scipy.spatial import distance_matrix

from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import (
    import_problem_module,
    installed_module,
    load_program_module,
    local_problem_environment,
    resolve_callable,
)

MAX_INSTANCES = 5
POSSIBLE_NAMES = ("select_next_node", "select_next_node_v1", "select_next_node_v2", "select_next_node_v3")
FULL_TEST_SIZES = {
    "val": (20, 50, 100, 200),
    "test": (20, 50, 100, 200, 500, 1000),
}


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {
            "combined_score": float(raw),
            "eval_time": 0.0,
            "error": None,
        }
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def _eval_instance(select_next_node: Callable[..., Any], node_positions: np.ndarray) -> float:
    problem_size = node_positions.shape[0]
    dist_mat = distance_matrix(node_positions, node_positions)
    start_node = 0
    solution = [start_node]
    unvisited = set(range(problem_size))
    unvisited.remove(start_node)

    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")

    return float(sum(dist_mat[solution[i], solution[(i + 1) % problem_size]] for i in range(problem_size)))


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="tsp_constructive_candidate")
        select_next_node = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("tsp_constructive")
        ds = base / "dataset" / "train50_dataset.npy"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {ds}，请在 task_assets/problems/tsp_constructive 下运行 gen_inst.generate_datasets()")
        nodes = np.load(ds)
        k = min(MAX_INSTANCES, len(nodes))
        objs = [_eval_instance(select_next_node, nodes[i]) for i in range(k)]
        return -float(np.mean(objs))

    return _wrap(inner)


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        if mode not in FULL_TEST_SIZES:
            raise ValueError(f"Unsupported full test mode: {mode}")
        module = load_program_module(program_code, module_name="tsp_constructive_full_candidate")
        select_next_node = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("tsp_constructive")
        size_to_objective: Dict[str, Dict[str, float]] = {}
        for problem_size in FULL_TEST_SIZES[mode]:
            ds = base / "dataset" / f"{mode}{problem_size}_dataset.npy"
            if not ds.is_file():
                gen_inst_mod = import_problem_module(base, "gen_inst")
                gen_inst_mod.generate_datasets()
            if not ds.is_file():
                raise FileNotFoundError(f"缺少数据集 {ds}")
            nodes = np.load(ds)
            objs = [_eval_instance(select_next_node, nodes[i]) for i in range(len(nodes))]
            avg_obj = float(np.mean(objs))
            size_to_objective[str(problem_size)] = {
                "objective": avg_obj,
                "combined_score": -avg_obj,
            }
        mean_combined_score = float(np.mean([x["combined_score"] for x in size_to_objective.values()]))
        return {
            "mode": mode,
            "problem_sizes": size_to_objective,
            "mean_combined_score": mean_combined_score,
            "error": None,
        }
    except Exception as e:
        return {"mode": mode, "problem_sizes": {}, "mean_combined_score": None, "error": str(e)}


def run_additional_test(program_code: str, *, label: str = "tsplib") -> Dict[str, Any]:
    try:
        if label != "tsplib":
            raise ValueError(f"Unsupported additional test label: {label}")
        candidate_module = load_program_module(program_code, module_name="tsp_constructive_tsplib_candidate")
        selector = resolve_callable(candidate_module, POSSIBLE_NAMES)
        if not hasattr(candidate_module, "select_next_node"):
            candidate_module.select_next_node = selector
        if not hasattr(candidate_module, "select_next_node_v2"):
            candidate_module.select_next_node_v2 = selector
        test_dir = problem_dir("tsp_constructive") / "test"
        with local_problem_environment(test_dir), installed_module("gpt", candidate_module):
            tsplib_mod = import_problem_module(test_dir, "test_tsplib")
            payload = tsplib_mod.run_tsplib_test(show_progress=False)
        payload["label"] = label
        return payload
    except Exception as e:
        return {"label": label, "instances": {}, "mean_gap_percent": None, "error": str(e)}

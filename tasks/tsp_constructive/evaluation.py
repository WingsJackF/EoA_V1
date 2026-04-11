"""任务专用评估：TSP constructive。"""

from __future__ import annotations

from copy import copy
from typing import Any, Callable, Dict

import numpy as np
from scipy.spatial import distance_matrix

from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import load_program_module, resolve_callable

MAX_INSTANCES = 5
POSSIBLE_NAMES = ("select_next_node", "select_next_node_v1", "select_next_node_v2", "select_next_node_v3")


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

"""任务专用评估：Online BPP priority。"""

from __future__ import annotations

import pickle
from typing import Any, Callable, Dict

import numpy as np

from tasks.task_support.paths import problem_dir
from tasks.task_support.runtime import load_program_module, resolve_callable

POSSIBLE_NAMES = ("priority", "priority_v1", "priority_v2", "priority_v3")


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def _get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    return np.nonzero((bins - item) >= 0)[0]


def _online_binpack(priority: Callable[..., Any], items: np.ndarray, bins: np.ndarray) -> np.ndarray:
    for item in items:
        valid_bin_indices = _get_valid_bin_indices(float(item), bins)
        priorities = priority(float(item), bins[valid_bin_indices])
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
    return bins


def _evaluate_dataset(priority: Callable[..., Any], instances: dict) -> float:
    num_bins = []
    for name, instance in instances.items():
        if name == "l1_bound":
            continue
        capacity = instance["capacity"]
        items = np.array(instance["items"]) if isinstance(instance["items"], list) else instance["items"]
        bins = np.array([capacity for _ in range(instance["num_items"])], dtype=float)
        packed = _online_binpack(priority, items.astype(float), bins)
        num_bins.append(int((packed != capacity).sum()))
    return float(np.mean(num_bins))


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="bpp_online_candidate")
        priority = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("bpp_online")
        ds = base / "dataset" / "weibull_5k_train.pickle"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {ds}，请在 task_assets/problems/bpp_online 下运行 gen_inst.generate_datasets()")
        with open(ds, "rb") as f:
            dataset = pickle.load(f)
        return _evaluate_dataset(priority, dataset)

    return _wrap(inner)

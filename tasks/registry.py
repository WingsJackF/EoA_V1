from __future__ import annotations

import importlib
import re
from typing import Callable, Dict, List

from tasks.base import EvolutionTask
from tasks.min_max_layout_16.task import build_min_max_layout_16_task
from tasks.tsp_constructive.task import build_tsp_constructive_task
from tasks.tsp_aco.task import build_tsp_aco_task
from tasks.tsp_gls.task import build_tsp_gls_task
from tasks.cvrp_aco.task import build_cvrp_aco_task
from tasks.op_aco.task import build_op_aco_task
from tasks.mkp_aco.task import build_mkp_aco_task
from tasks.bpp_offline_aco.task import build_bpp_offline_aco_task
from tasks.bpp_online.task import build_bpp_online_task
from tasks.dpp_ga.task import build_dpp_ga_task
from tasks.tsp_pomo.task import build_tsp_pomo_task
from tasks.tsp_lehd.task import build_tsp_lehd_task
from tasks.cvrp_pomo.task import build_cvrp_pomo_task
from tasks.cvrp_lehd.task import build_cvrp_lehd_task

TaskFactory = Callable[[], EvolutionTask]

TASK_REGISTRY: Dict[str, TaskFactory] = {
    "min_max_layout_16": build_min_max_layout_16_task,
    "tsp_constructive": build_tsp_constructive_task,
    "tsp_aco": build_tsp_aco_task,
    "tsp_gls": build_tsp_gls_task,
    "cvrp_aco": build_cvrp_aco_task,
    "op_aco": build_op_aco_task,
    "mkp_aco": build_mkp_aco_task,
    "bpp_offline_aco": build_bpp_offline_aco_task,
    "bpp_online": build_bpp_online_task,
    "dpp_ga": build_dpp_ga_task,
    "tsp_pomo": build_tsp_pomo_task,
    "tsp_lehd": build_tsp_lehd_task,
    "cvrp_pomo": build_cvrp_pomo_task,
    "cvrp_lehd": build_cvrp_lehd_task,
}


def get_task(task_id: str) -> EvolutionTask:
    if task_id not in TASK_REGISTRY:
        dynamic_factory = _load_generated_task(task_id)
        if dynamic_factory is not None:
            TASK_REGISTRY[task_id] = dynamic_factory
            return dynamic_factory()
        known = ", ".join(sorted(TASK_REGISTRY)) or "(empty)"
        raise KeyError(f"Unknown task_id={task_id!r}. Registered: {known}. Generated task folder `gen_task/{task_id}` was not loadable.")
    return TASK_REGISTRY[task_id]()


def list_tasks() -> List[str]:
    generated = _list_generated_tasks()
    return sorted(set(TASK_REGISTRY.keys()) | set(generated))


def register_task(task_id: str, factory: TaskFactory) -> None:
    """运行时注册新任务（例如插件加载）。"""
    TASK_REGISTRY[task_id] = factory


def _safe_task_id(task_id: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", task_id or ""))


def _load_generated_task(task_id: str) -> TaskFactory | None:
    """从 EoA_V1/gen_task/<task_id>/task.py 动态加载生成任务。"""
    if not _safe_task_id(task_id):
        return None
    try:
        module = importlib.import_module(f"gen_task.{task_id}.task")
    except ImportError:
        return None

    specific_name = f"build_{task_id}_task"
    factory = getattr(module, specific_name, None) or getattr(module, "build_task", None)
    return factory if callable(factory) else None


def _list_generated_tasks() -> List[str]:
    try:
        import gen_task
    except ImportError:
        return []
    root = getattr(gen_task, "__path__", None)
    if not root:
        return []
    tasks: List[str] = []
    for base in root:
        from pathlib import Path
        base_path = Path(base)
        for child in base_path.iterdir():
            if child.is_dir() and _safe_task_id(child.name) and (child / "task.py").exists():
                tasks.append(child.name)
    return tasks

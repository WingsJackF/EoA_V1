from __future__ import annotations

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
        known = ", ".join(sorted(TASK_REGISTRY)) or "(empty)"
        raise KeyError(f"Unknown task_id={task_id!r}. Registered: {known}")
    return TASK_REGISTRY[task_id]()


def list_tasks() -> List[str]:
    return sorted(TASK_REGISTRY.keys())


def register_task(task_id: str, factory: TaskFactory) -> None:
    """运行时注册新任务（例如插件加载）。"""
    TASK_REGISTRY[task_id] = factory

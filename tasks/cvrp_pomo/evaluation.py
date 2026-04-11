"""任务专用评估：CVRP POMO。"""

from __future__ import annotations

from typing import Any, Callable, Dict

from tasks.task_support.paths import problem_dir


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"min_max_ratio": 0.0, "combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"min_max_ratio": 0.0, "combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        p = problem_dir("cvrp_pomo")
        ck = p / "checkpoints"
        if not ck.is_dir() or not any(ck.iterdir()):
            raise RuntimeError("需要预训练权重与数据，请将 checkpoint 放入 task_assets/problems/cvrp_pomo/checkpoints")
        raise RuntimeError("CVRP POMO 的完整测试管线尚未在当前项目中内联实现。")

    return _wrap(inner)

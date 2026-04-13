"""Spawn 子进程跑单次评测并支持墙钟超时（可 terminate 卡住 CUDA 的 worker）。"""

from __future__ import annotations

import math
import queue
import time
import multiprocessing as mp
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from tasks.base import FAILED_COMBINED_SCORE
from tasks.task_support.processes import cleanup_process_pool


def eval_timeout_is_disabled(evaluation_timeout_seconds: float | None) -> bool:
    if evaluation_timeout_seconds is None:
        return True
    try:
        t = float(evaluation_timeout_seconds)
    except (TypeError, ValueError):
        return True
    if math.isnan(t) or t <= 0.0 or math.isinf(t):
        return True
    return False


def _subprocess_missing_result_payload() -> Dict[str, Any]:
    return {
        "individual": {
            "thought": "",
            "code": "",
            "fitness": {
                "combined_score": FAILED_COMBINED_SCORE,
                "eval_time": 0.0,
                "error": "Evaluation subprocess exited without returning a result",
            },
        },
        "elapsed": 0.0,
    }


def timeout_failure_payload(limit_seconds: float) -> Dict[str, Any]:
    lim = float(limit_seconds)
    return {
        "individual": {
            "thought": "",
            "code": "",
            "fitness": {
                "combined_score": FAILED_COMBINED_SCORE,
                "eval_time": lim,
                "error": f"Evaluation exceeded time limit ({lim:g} s)",
            },
        },
        "elapsed": lim,
    }


def spawn_offspring_eval_worker(
    task_id: str,
    raw_content: Any,
    logical_cuda_device: int | None,
    out_q: Any,
) -> None:
    from implement_evolutionary_operators_module.design_offspring_generation_controller import (
        _evaluate_single_offspring,
    )

    try:
        out_q.put(_evaluate_single_offspring(task_id, raw_content, logical_cuda_device))
    except Exception as exc:  # noqa: BLE001
        out_q.put(
            {
                "individual": {
                    "thought": "",
                    "code": "",
                    "fitness": {
                        "combined_score": FAILED_COMBINED_SCORE,
                        "eval_time": 0.0,
                        "error": f"Evaluator worker crashed: {exc!r}",
                    },
                },
                "elapsed": 0.0,
            }
        )


def spawn_initial_eval_worker(
    task_id: str,
    raw_content: str,
    logical_cuda_device: int | None,
    out_q: Any,
) -> None:
    from develop_population_manager_module.initialize_population_module import _evaluate_initial_raw_content

    try:
        out_q.put(_evaluate_initial_raw_content(task_id, raw_content, logical_cuda_device))
    except Exception as exc:  # noqa: BLE001
        out_q.put(
            {
                "individual": {
                    "thought": "",
                    "code": "",
                    "fitness": {
                        "combined_score": FAILED_COMBINED_SCORE,
                        "eval_time": 0.0,
                        "error": f"Evaluator worker crashed: {exc!r}",
                    },
                },
                "elapsed": 0.0,
            }
        )


def run_spawn_eval_jobs(
    jobs: List[Tuple[int, Tuple[Any, ...]]],
    *,
    max_workers: int,
    job_timeout_seconds: float,
    worker_target: Callable[..., None],
    on_job_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    对每个 job 启动独立 spawn 子进程，限制并发为 max_workers，单任务墙钟超时后 terminate。

    jobs: (结果下标, (传给 worker_target 的参数元组，不含末尾的 Queue))。
    worker_target 签名末尾为 out_q。
    on_job_complete: 每完成一项（正常结束、超时或子进程异常退出）时调用，便于立即打日志。
    """
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if job_timeout_seconds <= 0:
        raise ValueError("job_timeout_seconds must be positive")

    ctx = mp.get_context("spawn")
    pending: deque[Tuple[int, Tuple[Any, ...]]] = deque(jobs)
    running: List[Tuple[mp.Process, Any, int, float]] = []
    results: Dict[int, Dict[str, Any]] = {}

    def _finish_job(idx: int, payload: Dict[str, Any]) -> None:
        results[idx] = payload
        if on_job_complete is not None:
            on_job_complete(idx, payload)

    def _start_one() -> None:
        if len(running) >= max_workers or not pending:
            return
        idx, job_args = pending.popleft()
        out_q = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=worker_target, args=job_args + (out_q,))
        proc.start()
        running.append((proc, out_q, idx, time.perf_counter()))

    try:
        while pending or running:
            while len(running) < max_workers and pending:
                _start_one()
            now = time.perf_counter()
            for entry in list(running):
                proc, out_q, idx, t0 = entry
                if not proc.is_alive():
                    running.remove(entry)
                    try:
                        payload = out_q.get_nowait()
                    except queue.Empty:
                        payload = _subprocess_missing_result_payload()
                    _finish_job(idx, payload)
                elif (now - t0) >= job_timeout_seconds:
                    running.remove(entry)
                    proc.terminate()
                    proc.join(2.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(1.0)
                    try:
                        out_q.get_nowait()
                    except queue.Empty:
                        pass
                    _finish_job(idx, timeout_failure_payload(job_timeout_seconds))
            if running:
                time.sleep(0.05)
    except KeyboardInterrupt:
        for proc, _q, _idx, _t0 in running:
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(0.5)
        cleanup_process_pool(None)
        raise

    return results

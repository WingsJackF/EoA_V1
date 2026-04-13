"""Helpers for cleaning up multiprocessing workers on interruption."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any


def _stop_process(proc: Any) -> None:
    if proc is None:
        return
    try:
        if not proc.is_alive():
            return
    except Exception:
        return
    try:
        proc.terminate()
        proc.join(timeout=2.0)
    except Exception:
        pass
    try:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
    except Exception:
        pass


def cleanup_process_pool(executor: Any | None = None) -> None:
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        for proc in getattr(executor, "_processes", {}).values():
            _stop_process(proc)

    for child in mp.active_children():
        _stop_process(child)

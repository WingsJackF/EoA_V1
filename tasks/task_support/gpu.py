"""GPU runtime helpers shared by CLI and task evaluators."""

from __future__ import annotations

import os
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def gpu_requested() -> bool:
    return _env_flag("EOA_USE_CUDA")


def logical_cuda_device_num() -> int:
    raw = os.environ.get("EOA_CUDA_DEVICE_NUM", "0").strip() or "0"
    return int(raw)


def visible_gpu() -> str:
    return os.environ.get("EOA_VISIBLE_GPU", "").strip()


def solver_device() -> str:
    if not gpu_requested():
        return "cpu"
    return f"cuda:{logical_cuda_device_num()}"


def configure_gpu_environment(gpu_index: int | None) -> None:
    if gpu_index is None:
        os.environ.pop("EOA_USE_CUDA", None)
        os.environ.pop("EOA_VISIBLE_GPU", None)
        os.environ.pop("EOA_CUDA_DEVICE_NUM", None)
        return
    if gpu_index < 0:
        raise ValueError("--gpu must be a non-negative integer")
    os.environ["EOA_USE_CUDA"] = "1"
    os.environ["EOA_VISIBLE_GPU"] = str(gpu_index)
    os.environ["EOA_CUDA_DEVICE_NUM"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def apply_module_gpu_overrides(module: Any) -> None:
    use_cuda = gpu_requested()
    cuda_device_num = logical_cuda_device_num()
    if hasattr(module, "USE_CUDA"):
        setattr(module, "USE_CUDA", use_cuda)
    if hasattr(module, "CUDA_DEVICE_NUM"):
        setattr(module, "CUDA_DEVICE_NUM", cuda_device_num)
    for attr in ("tester_params", "trainer_params"):
        params = getattr(module, attr, None)
        if isinstance(params, dict):
            if "use_cuda" in params:
                params["use_cuda"] = use_cuda
            if "cuda_device_num" in params:
                params["cuda_device_num"] = cuda_device_num

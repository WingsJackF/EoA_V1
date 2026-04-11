"""定位 `EoA` 内置任务资产目录。"""

from pathlib import Path


def exp_root() -> Path:
    return Path(__file__).resolve().parents[3]


def eoa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def task_assets_root() -> Path:
    return eoa_root() / "task_assets"


def problems_root() -> Path:
    return task_assets_root() / "problems"


def prompts_root() -> Path:
    return task_assets_root() / "prompts"


def problem_dir(name: str) -> Path:
    return problems_root() / name


def prompt_dir(name: str) -> Path:
    return prompts_root() / name

"""任务运行时辅助：加载候选程序、按问题目录导入局部模块。"""

from __future__ import annotations

import importlib
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator


def load_program_module(program_code: str, *, module_name: str = "_candidate_program") -> types.ModuleType:
    if not isinstance(program_code, str):
        raise TypeError("program_code must be a string")
    module = types.ModuleType(module_name)
    exec(compile(program_code, f"<{module_name}>", "exec"), module.__dict__)
    return module


def resolve_callable(module: types.ModuleType, candidate_names: Iterable[str]) -> Callable[..., object]:
    for name in candidate_names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    raise AttributeError(f"No callable found among: {', '.join(candidate_names)}")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def purge_modules_under(root: Path) -> None:
    resolved_root = root.resolve()
    for name, module in list(sys.modules.items()):
        file_attr = getattr(module, "__file__", None)
        if not file_attr:
            continue
        try:
            module_path = Path(file_attr).resolve()
        except OSError:
            continue
        if _is_under(module_path, resolved_root):
            sys.modules.pop(name, None)


@contextmanager
def local_problem_environment(problem_path: Path) -> Iterator[None]:
    inserted = False
    problem_path = problem_path.resolve()
    if str(problem_path) not in sys.path:
        sys.path.insert(0, str(problem_path))
        inserted = True
    old_cwd = os.getcwd()
    os.chdir(problem_path)
    try:
        purge_modules_under(problem_path)
        yield
    finally:
        os.chdir(old_cwd)
        purge_modules_under(problem_path)
        if inserted:
            try:
                sys.path.remove(str(problem_path))
            except ValueError:
                pass


@contextmanager
def installed_module(name: str, module: types.ModuleType) -> Iterator[None]:
    old = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def import_problem_module(problem_path: Path, module_name: str):
    with local_problem_environment(problem_path):
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

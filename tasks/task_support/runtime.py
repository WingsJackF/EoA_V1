"""任务运行时辅助：加载候选程序、按问题目录导入局部模块。"""

from __future__ import annotations

import importlib
import builtins
import io
import os
import shutil
import socket
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


class SandboxViolation(PermissionError):
    """Raised when candidate code attempts an operation outside the eval sandbox."""


def _default_allowed_read_roots() -> tuple[Path, ...]:
    eoa_root = Path(__file__).resolve().parents[2]
    problems_root = eoa_root / "task_assets" / "problems"
    allowed_names = {
        "dataset",
        "datasets",
        "data",
        "checkpoints",
        "test_problems",
        "DPP_data",
        "Transform_data",
        "tsplib",
    }
    roots: list[Path] = []
    if problems_root.is_dir():
        for child in problems_root.rglob("*"):
            if child.is_dir() and child.name in allowed_names:
                try:
                    roots.append(child.resolve())
                except OSError:
                    continue
    return tuple(roots)


def _normalize_allowed_read_roots(allowed_read_roots: Sequence[str | Path] | None) -> tuple[Path, ...]:
    roots: list[Path] = []
    values: list[str | Path] = list(_default_allowed_read_roots())
    if allowed_read_roots:
        values.extend(allowed_read_roots)
    env_roots = os.environ.get("EOA_SANDBOX_READ_ROOTS") or os.environ.get("EVAL_SANDBOX_READ_ROOTS")
    if env_roots:
        values.extend(part for part in env_roots.split(os.pathsep) if part.strip())
    for value in values:
        try:
            root = Path(value).expanduser().resolve()
        except OSError:
            continue
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _is_path_allowed(path: str | os.PathLike[str], allowed_roots: tuple[Path, ...]) -> bool:
    try:
        target = Path(path).expanduser().resolve()
    except OSError:
        return False
    return any(_is_under(target, root) for root in allowed_roots)


def _check_read_path(path: Any, allowed_roots: tuple[Path, ...]) -> None:
    if isinstance(path, int):
        raise SandboxViolation("Candidate sandbox blocks file-descriptor based open().")
    if not _is_path_allowed(path, allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots) or "(none)"
        raise SandboxViolation(f"Candidate sandbox blocked file read: {path!r}; allowed read roots: {allowed}")


def _check_read_mode(mode: str) -> None:
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        raise SandboxViolation(f"Candidate sandbox blocks file write mode: {mode!r}")


class _SandboxedCallable:
    def __init__(self, func: Callable[..., object], allowed_read_roots: tuple[Path, ...]) -> None:
        self._func = func
        self._allowed_read_roots = allowed_read_roots
        self.__name__ = getattr(func, "__name__", self.__class__.__name__)
        self.__doc__ = getattr(func, "__doc__", None)
        self.__module__ = getattr(func, "__module__", None)

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        with candidate_sandbox(self._allowed_read_roots):
            return self._func(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._func, name)


@contextmanager
def candidate_sandbox(allowed_read_roots: Sequence[str | Path] | None = None) -> Iterator[None]:
    """
    Best-effort Python-level sandbox for LLM-generated candidate code.

    It blocks filesystem writes, subprocess creation, sockets, and arbitrary file
    reads except under EOA_SANDBOX_READ_ROOTS / EVAL_SANDBOX_READ_ROOTS or the
    explicit allowed_read_roots argument. This is not an OS security boundary.
    """
    roots = _normalize_allowed_read_roots(allowed_read_roots)

    orig_open = builtins.open
    orig_io_open = io.open
    orig_path_open = Path.open
    orig_path_read_text = Path.read_text
    orig_path_read_bytes = Path.read_bytes
    orig_import = builtins.__import__

    os_patches = {
        "open": getattr(os, "open", None),
        "remove": getattr(os, "remove", None),
        "unlink": getattr(os, "unlink", None),
        "rename": getattr(os, "rename", None),
        "replace": getattr(os, "replace", None),
        "rmdir": getattr(os, "rmdir", None),
        "removedirs": getattr(os, "removedirs", None),
        "mkdir": getattr(os, "mkdir", None),
        "makedirs": getattr(os, "makedirs", None),
        "chmod": getattr(os, "chmod", None),
        "chown": getattr(os, "chown", None),
        "link": getattr(os, "link", None),
        "symlink": getattr(os, "symlink", None),
        "system": getattr(os, "system", None),
        "popen": getattr(os, "popen", None),
        "fork": getattr(os, "fork", None),
    }
    subprocess_patches = {
        "Popen": subprocess.Popen,
        "run": subprocess.run,
        "call": subprocess.call,
        "check_call": subprocess.check_call,
        "check_output": subprocess.check_output,
    }
    socket_patches = {
        "socket": socket.socket,
        "create_connection": socket.create_connection,
    }
    shutil_patches = {
        "rmtree": shutil.rmtree,
        "copy": shutil.copy,
        "copy2": shutil.copy2,
        "copyfile": shutil.copyfile,
        "move": shutil.move,
    }

    blocked_import_roots = {
        "subprocess",
        "socket",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "telnetlib",
        "requests",
    }

    def sandboxed_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        _check_read_mode(mode)
        _check_read_path(file, roots)
        return orig_open(file, mode, *args, **kwargs)

    def sandboxed_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        _check_read_mode(mode)
        _check_read_path(file, roots)
        return orig_io_open(file, mode, *args, **kwargs)

    def sandboxed_path_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        _check_read_mode(mode)
        _check_read_path(self, roots)
        return orig_path_open(self, mode, *args, **kwargs)

    def sandboxed_path_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        _check_read_path(self, roots)
        return orig_path_read_text(self, *args, **kwargs)

    def sandboxed_path_read_bytes(self: Path) -> bytes:
        _check_read_path(self, roots)
        return orig_path_read_bytes(self)

    def blocked_operation(*_args: Any, **_kwargs: Any) -> Any:
        raise SandboxViolation("Candidate sandbox blocked filesystem/process/network operation.")

    def sandboxed_import(name: str, globals: Any = None, locals: Any = None, fromlist: tuple = (), level: int = 0) -> Any:
        root = name.split(".", 1)[0]
        if level == 0 and root in blocked_import_roots:
            raise SandboxViolation(f"Candidate sandbox blocked import: {name}")
        return orig_import(name, globals, locals, fromlist, level)

    builtins.open = sandboxed_open
    io.open = sandboxed_io_open
    Path.open = sandboxed_path_open
    Path.read_text = sandboxed_path_read_text
    Path.read_bytes = sandboxed_path_read_bytes
    builtins.__import__ = sandboxed_import
    try:
        for name in os_patches:
            if os_patches[name] is not None:
                setattr(os, name, blocked_operation)
        for name in subprocess_patches:
            setattr(subprocess, name, blocked_operation)
        for name in socket_patches:
            setattr(socket, name, blocked_operation)
        for name in shutil_patches:
            setattr(shutil, name, blocked_operation)
        yield
    finally:
        builtins.open = orig_open
        io.open = orig_io_open
        Path.open = orig_path_open
        Path.read_text = orig_path_read_text
        Path.read_bytes = orig_path_read_bytes
        builtins.__import__ = orig_import
        for name, value in os_patches.items():
            if value is not None:
                setattr(os, name, value)
        for name, value in subprocess_patches.items():
            setattr(subprocess, name, value)
        for name, value in socket_patches.items():
            setattr(socket, name, value)
        for name, value in shutil_patches.items():
            setattr(shutil, name, value)


def load_program_module(
    program_code: str,
    *,
    module_name: str = "_candidate_program",
    allowed_read_roots: Sequence[str | Path] | None = None,
    sandbox: bool = True,
) -> types.ModuleType:
    if not isinstance(program_code, str):
        raise TypeError("program_code must be a string")
    module = types.ModuleType(module_name)
    roots = _normalize_allowed_read_roots(allowed_read_roots)
    module.__dict__["__candidate_sandbox_read_roots__"] = roots
    if sandbox:
        with candidate_sandbox(roots):
            exec(compile(program_code, f"<{module_name}>", "exec"), module.__dict__)
    else:
        exec(compile(program_code, f"<{module_name}>", "exec"), module.__dict__)
    return module


def resolve_callable(module: types.ModuleType, candidate_names: Iterable[str]) -> Callable[..., object]:
    if isinstance(candidate_names, str):
        names = (candidate_names,)
    else:
        names = tuple(candidate_names)
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            roots = getattr(module, "__candidate_sandbox_read_roots__", None)
            if roots is not None:
                return _SandboxedCallable(obj, roots)
            return obj
    joined_names = ", ".join(names) if names else "<none>"
    raise AttributeError(f"No callable found among: {joined_names}")


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

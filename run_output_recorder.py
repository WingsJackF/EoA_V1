"""
单次运行输出目录：output/<task_id>/<时间戳>/terminal.log + evolution/*.json + meta。
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO


def make_run_output_dir(eoa_root: Path, task_id: str) -> Path:
    # 含微秒，避免同一秒内多次运行目录冲突
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = eoa_root / "output" / task_id / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evolution").mkdir(exist_ok=True)
    return run_dir


class _TeeBuffer:
    def __init__(self, orig_buffer: Any, log_text_io: TextIO) -> None:
        self._orig = orig_buffer
        self._log = log_text_io

    def write(self, b: bytes) -> int:
        n = self._orig.write(b)
        self._log.write(b.decode("utf-8", errors="replace"))
        self._log.flush()
        return n

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class TeeTextIO:
    """将文本 stdout/stderr 同时写入原始流与日志文件（含 buffer 回退路径）。"""

    def __init__(self, stream: TextIO, log_f: TextIO) -> None:
        self.stream = stream
        self.log_f = log_f
        self._buffer: Optional[_TeeBuffer] = None

    def write(self, s: str) -> int:
        n = self.stream.write(s)
        self.log_f.write(s)
        self.log_f.flush()
        return n

    def flush(self) -> None:
        self.stream.flush()
        self.log_f.flush()

    @property
    def buffer(self) -> _TeeBuffer:
        if self._buffer is None:
            self._buffer = _TeeBuffer(self.stream.buffer, self.log_f)
        return self._buffer

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


@contextmanager
def tee_terminal_to_file(run_dir: Path) -> Iterator[None]:
    log_path = run_dir / "terminal.log"
    old_out, old_err = sys.stdout, sys.stderr
    f = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        sys.stdout = TeeTextIO(old_out, f)
        sys.stderr = TeeTextIO(old_err, f)
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()


def write_run_meta(
    run_dir: Path,
    *,
    task_id: str,
    target_function_name: str,
    argv: List[str],
    llm_meta: Dict[str, Any],
) -> None:
    payload = {
        "task_id": task_id,
        "target_function_name": target_function_name,
        "run_directory": str(run_dir.resolve()),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "argv": argv,
        "llm": llm_meta,
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _jsonable_individual(ind: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "thought": ind.get("thought", ""),
        "code": ind.get("code", ""),
        "fitness": ind.get("fitness", {}),
    }


def save_generation_snapshot(
    run_dir: Path,
    *,
    generation: int,
    label: str,
    population: List[Dict[str, Any]],
    archive: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    data: Dict[str, Any] = {
        "generation": generation,
        "label": label,
        "population": [_jsonable_individual(x) for x in population],
        "archive": [_jsonable_individual(x) for x in archive],
    }
    if extra:
        data["extra"] = extra
    path = run_dir / "evolution" / f"generation_{generation:03d}_{label}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_final_archive(run_dir: Path, archive: List[Dict[str, Any]]) -> None:
    path = run_dir / "final_archive.json"
    path.write_text(
        json.dumps([_jsonable_individual(x) for x in archive], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

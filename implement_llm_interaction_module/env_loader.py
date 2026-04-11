from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

_LOADED_ENV_PATHS: set[Path] = set()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_env_paths() -> Iterable[Path]:
    override = os.environ.get("EOA_ENV_FILE", "").strip()
    if override:
        yield Path(override).expanduser()
    yield project_root() / ".env"
    yield Path.cwd() / ".env"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_quotes(value)
    return values


def load_project_env(*, override_existing: bool = False) -> None:
    """
    Load project-level ``.env`` values into ``os.environ`` without requiring python-dotenv.

    Existing process environment variables win by default so shell/CI secrets can override
    local files. Set ``EOA_ENV_FILE`` to point at a custom env file.
    """
    for path in _candidate_env_paths():
        resolved = path.resolve()
        if resolved in _LOADED_ENV_PATHS or not resolved.is_file():
            continue
        for key, value in _parse_env_file(resolved).items():
            if override_existing or os.environ.get(key) in (None, ""):
                os.environ[key] = value
        _LOADED_ENV_PATHS.add(resolved)


def get_env(name: str, default: str = "", *, aliases: Optional[Iterable[str]] = None) -> str:
    load_project_env()
    for key in (name, *(aliases or ())):
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return default

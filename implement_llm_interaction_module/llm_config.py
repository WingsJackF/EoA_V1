"""
LLM 配置器：统一「远程 API」与「本地 OpenAI 兼容服务」（Ollama / LM Studio / vLLM 等）。

统一走 HTTP ``POST {base_url}/chat/completions``，与 OpenAI SDK 兼容的请求体。

## 配置来源（优先级：CLI 覆盖 > `.env` / 进程环境变量 > JSON 兼容配置 > 预设）

- ``.env``：项目根目录下的本地配置文件（推荐，默认自动加载；不要提交到 Git）
- ``EOA_ENV_FILE``：可选，指定自定义 `.env` 路径
- ``LLM_PRESET``：``ollama`` | ``openai`` | ``deepseek`` | ``agicto`` | ``vllm_qwen3`` 等（不设则用下面各项或内置默认）
- ``LLM_BASE_URL``：API 根路径，**不含** ``/chat/completions``。例：``https://api.openai.com/v1``、``http://127.0.0.1:11434/v1``
- ``LLM_API_KEY``：Bearer Token；本地 Ollama 可填 ``ollama`` 或任意非空占位
- ``LLM_MODEL``：模型名
- ``LLM_TIMEOUT``：单次请求超时秒数（浮点）
- ``LLM_CONFIG_FILE``：可选 JSON 配置文件路径（兼容旧配置，不建议放密钥）

## JSON 配置示例（``LLM_CONFIG_FILE``）

.. code-block:: json

    {
      "base_url": "http://127.0.0.1:1234/v1",
      "api_key": "not-needed",
      "model": "mistral-7b",
      "timeout": 120.0
    }

## 预设

- **ollama**：``http://127.0.0.1:11434/v1``，默认模型 ``llama3.2``（可用 ``LLM_MODEL`` 覆盖）
- **openai**：``https://api.openai.com/v1``，必须设置 ``LLM_API_KEY``
- **deepseek**：``https://api.deepseek.com/v1``，默认模型 ``deepseek-chat``（请设置 ``LLM_API_KEY``）
- **vllm_qwen3**：本机 vLLM ``http://127.0.0.1:10008/v1``，``model`` 与 ``GET /v1/models`` 返回的 ``id`` 一致（常为本地路径；换机请设 ``LLM_MODEL``）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from implement_llm_interaction_module.env_loader import get_env, load_project_env

__all__ = [
    "LLMSettings",
    "chat_completions_url",
    "clear_llm_settings_cache",
    "configure_llm",
    "get_llm_settings",
    "load_llm_settings",
    "load_project_env",
    "request_headers",
]


@dataclass(frozen=True)
class LLMSettings:
    """单次请求所需的端点与鉴权（OpenAI 兼容）。"""

    base_url: str
    """例如 ``https://api.openai.com/v1`` 或 ``http://127.0.0.1:11434/v1``（无末尾斜杠）。"""
    api_key: str
    model: str
    timeout: float = 120.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def provider_label(self) -> str:
        if "127.0.0.1" in self.base_url or "localhost" in self.base_url:
            return "local_http"
        return "api"


def chat_completions_url(settings: LLMSettings) -> str:
    return settings.base_url.rstrip("/") + "/chat/completions"


def request_headers(settings: LLMSettings) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if settings.api_key:
        h["Authorization"] = f"Bearer {settings.api_key}"
    h.update(dict(settings.extra_headers))
    return h


_PRESETS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "model": "llama3.2",
        "timeout": 120.0,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "timeout": 120.0,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-chat",
        "timeout": 120.0,
    },
    # 与旧代码兼容的第三方网关默认基址（密钥必须通过环境变量提供）
    "agicto": {
        "base_url": "https://api.agicto.cn/v1",
        "api_key": "",
        "model": "gpt-5-mini",
        "timeout": 120.0,
    },
    # 本机 vLLM：端口与模型名可按需用环境变量 LLM_BASE_URL / LLM_MODEL 覆盖
    "vllm_qwen3": {
        "base_url": "http://127.0.0.1:10008/v1",
        "api_key": "",
        "model": "/data2/jfwang/models/omni-3",
        "timeout": 300.0,
    },
}


def _env_str(name: str, default: str = "") -> str:
    # 兼容常见 OpenAI 环境名
    if name == "LLM_API_KEY":
        return get_env(name, default, aliases=("OPENAI_API_KEY",))
    if name == "LLM_BASE_URL":
        return get_env(name, default, aliases=("OPENAI_BASE_URL",))
    if name == "LLM_MODEL":
        return get_env(name, default, aliases=("OPENAI_MODEL",))
    return get_env(name, default)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _read_json_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_llm_settings(
    *,
    config_file: Optional[str] = None,
    preset: Optional[str] = None,
) -> LLMSettings:
    """
    从预设、``LLM_CONFIG_FILE`` / *config_file*、``.env`` / 环境变量构造配置（不读写全局缓存）。
    """
    load_project_env()
    cfg: Dict[str, Any] = {}

    pre = preset or _env_str("LLM_PRESET", "").lower()
    if pre and pre in _PRESETS:
        cfg.update(_PRESETS[pre])

    cf = config_file or _env_str("LLM_CONFIG_FILE", "")
    if cf:
        cfg.update(_read_json_config(Path(cf)))

    base_url = _env_str("LLM_BASE_URL", str(cfg.get("base_url", "")))
    api_key = _env_str("LLM_API_KEY", str(cfg.get("api_key", "")))
    model = _env_str("LLM_MODEL", str(cfg.get("model", "")))
    timeout = _env_float("LLM_TIMEOUT", float(cfg.get("timeout", 120.0)))

    extra = cfg.get("extra_headers")
    extra_headers: Dict[str, str] = dict(extra) if isinstance(extra, dict) else {}

    # 若仍未指定 base_url：默认走 deepseek 基址；本地请显式设 LLM_PRESET=ollama 或 LLM_BASE_URL
    if not base_url:
        base_url = _PRESETS["deepseek"]["base_url"]
    if not model:
        model = _PRESETS["deepseek"]["model"]

    return LLMSettings(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout=timeout,
        extra_headers=extra_headers,
    )


_settings_cache: Optional[LLMSettings] = None


def get_llm_settings() -> LLMSettings:
    """返回进程内缓存的配置；首次调用时从环境变量加载。"""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load_llm_settings()
    return _settings_cache


def clear_llm_settings_cache() -> None:
    global _settings_cache
    _settings_cache = None


def configure_llm(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    extra_headers: Optional[Mapping[str, str]] = None,
    replace_all: bool = False,
) -> LLMSettings:
    """
    在运行中覆盖配置（如 ``main`` 解析 CLI 之后调用）。

    - ``replace_all=True``：忽略当前缓存，从环境重新 ``load_llm_settings`` 再应用覆盖。
    """
    global _settings_cache
    if replace_all or _settings_cache is None:
        base = load_llm_settings()
    else:
        base = _settings_cache

    kwargs: Dict[str, Any] = {}
    if base_url is not None:
        kwargs["base_url"] = base_url.rstrip("/")
    if api_key is not None:
        kwargs["api_key"] = api_key
    if model is not None:
        kwargs["model"] = model
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    if extra_headers is not None:
        kwargs["extra_headers"] = {**dict(base.extra_headers), **dict(extra_headers)}

    _settings_cache = replace(base, **kwargs) if kwargs else base
    return _settings_cache

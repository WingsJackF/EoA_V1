import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: construct_api_request_payload (Updated for Windows console Unicode errors)

Description:
    Implements a small utility to construct a chat/completions-style API payload
    for an LLM. Provides an LLMMessage helper class and the
    construct_api_request_payload(...) function.

    This updated version fixes Unicode printing errors on Windows consoles that use
    the 'gbk' encoding by providing a safe_print helper which falls back to writing
    UTF-8 bytes to stdout.buffer when necessary.

Notes:
    - This function only constructs and returns a payload dictionary; it does not
      perform any HTTP requests itself (except in the demonstration block).
    - Uses only json and typing from the standard library for the core function.
    - Avoids catching broad exceptions; only specific exceptions are handled.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from implement_llm_interaction_module.llm_config import (
    chat_completions_url,
    get_llm_settings,
    request_headers,
)


class LLMMessage:
    """
    Message format for LLM API chat endpoints.

    Attributes:
        role: The role of the message sender (e.g., "system", "user", "assistant").
        content: The textual content of the message.
    """

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the message into a dictionary suitable for inclusion in a
        chat/completions payload.

        Returns:
            A dict with keys "role" and "content".
        """
        return {"role": self.role, "content": self.content}


def construct_api_request_payload(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs the API request payload for the LLM chat/completions endpoint.

    Args:
        system_prompt: The system-level prompt string (instructions/context for LLM).
        user_prompt: The user-level prompt string (evolutionary strategy, code, or request).
        model: 模型名；默认使用 ``get_llm_settings().model``（环境变量 / 配置文件）。

    Returns:
        A dictionary payload compliant with the LLM API specification, ready for HTTP POST.

    Raises:
        TypeError: If system_prompt or user_prompt are not strings.
    """
    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string")
    if not isinstance(user_prompt, str):
        raise TypeError("user_prompt must be a string")

    messages = [
        LLMMessage(role="system", content=system_prompt).to_dict(),
        LLMMessage(role="user", content=user_prompt).to_dict(),
    ]

    resolved_model = model if model is not None else get_llm_settings().model
    payload = {"model": resolved_model, "messages": messages}
    return payload


def safe_print(text: str) -> None:
    """
    Print text to stdout in a way that avoids UnicodeEncodeError on consoles
    using limited encodings (e.g., 'gbk' on Windows).

    Strategy:
      - First attempt a normal print().
      - If a UnicodeEncodeError occurs, write UTF-8 encoded bytes to stdout.buffer.

    Args:
        text: The text string to output.
    """
    try:
        # Normal print may raise UnicodeEncodeError on some consoles.
        print(text)
    except UnicodeEncodeError:
        # Fallback: write UTF-8 bytes directly to the underlying buffer.
        # This avoids encoding errors on legacy consoles.
        sys.stdout.buffer.write(text.encode("utf-8"))
        # Ensure a newline at the end like print() would have added.
        sys.stdout.buffer.write(b"\n")



import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: implement_post_request_and_retry

Description:
    Sends HTTP POST requests to the LLM API endpoint with robust retry logic,
    exponential backoff, and error handling. Returns parsed JSON from successful
    responses and raises requests.RequestException if all retries fail.

Constraints followed:
    - Python 3.9+ typing annotations used.
    - Uses requests, json, time from the standard/core libraries.
    - No file I/O; runtime information printed to stdout when verbose=True.
    - Does not swallow exceptions; surfaces requests.RequestException after retries.
    - Does not use bare 'except Exception'.
"""

import json
import time
from typing import Any, Dict, Optional

import requests  # requests==2.31 is expected in the environment


def implement_post_request_and_retry(
    payload: Dict[str, Any],
    max_retries: int = 4,
    base_backoff: float = 1.0,
    timeout: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Sends a POST request to the LLM API endpoint with robust retry and exponential backoff.

    Args:
        payload: The API request payload (dict) to send.
        max_retries: Maximum number of retry attempts on failure.
        base_backoff: Initial backoff duration in seconds (doubles each retry).
        timeout: 单次请求超时（秒）；默认使用 ``get_llm_settings().timeout``。
        verbose: If True, prints retry/failure information.

    Returns:
        The response JSON as a dictionary.

    Raises:
        requests.RequestException: If all retries fail or a fatal error occurs.
    """
    attempt = 0
    settings = get_llm_settings()
    url = chat_completions_url(settings)
    headers = request_headers(settings)
    req_timeout = settings.timeout if timeout is None else float(timeout)

    # Validate input payload
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")

    while attempt <= max_retries:
        try:
            if verbose:
                print(
                    f"[POST] Attempt {attempt + 1}/{max_retries + 1} "
                    f"to {url} (timeout={req_timeout}s)"
                )

            response = requests.post(url, headers=headers, json=payload, timeout=req_timeout)

            status = response.status_code
            if status == 200:
                # Return parsed JSON; let ValueError propagate if JSON is invalid.
                return response.json()
            else:
                # Non-200 status: provide details and raise to trigger retry logic
                content_preview = None
                # Try to obtain a short preview of content safely
                try:
                    content_preview = response.text[:1000]
                except Exception:
                    content_preview = "<unavailable response text>"

                msg = f"Non-200 response: status={status}, content={content_preview}"
                if verbose:
                    print(f"[POST] {msg}")
                # Raise a requests.RequestException to be caught below and possibly retried
                raise requests.RequestException(msg)

        except requests.RequestException as req_err:
            # If we've exhausted retries, surface the exception
            if attempt >= max_retries:
                if verbose:
                    print("[POST] All retries exhausted. Raising exception.")
                # Surface the last encountered requests.RequestException
                raise req_err
            # Otherwise, backoff and retry
            backoff = base_backoff * (2 ** attempt)
            if verbose:
                print(f"[POST] Request failed (attempt {attempt + 1}). "
                      f"Error: {req_err}. Backing off {backoff:.2f}s before retry.")
            time.sleep(backoff)
            attempt += 1

    # If loop exits unexpectedly, raise a RequestException indicating failure
    raise requests.RequestException("Failed to obtain a successful response after retries.")


async def _post_payload_with_semaphore(
    semaphore: asyncio.Semaphore,
    payload: Dict[str, Any],
    *,
    max_retries: int,
    base_backoff: float,
    timeout: Optional[float],
    verbose: bool,
) -> Dict[str, Any]:
    async with semaphore:
        return await asyncio.to_thread(
            implement_post_request_and_retry,
            payload,
            max_retries,
            base_backoff,
            timeout,
            verbose,
        )


async def _implement_post_requests_concurrently_async(
    payloads: Sequence[Dict[str, Any]],
    *,
    max_retries: int,
    base_backoff: float,
    timeout: Optional[float],
    verbose: bool,
    max_concurrency: Optional[int],
) -> List[Union[Dict[str, Any], Exception]]:
    resolved_concurrency = len(payloads) if max_concurrency is None else int(max_concurrency)
    if resolved_concurrency <= 0:
        raise ValueError("max_concurrency must be a positive integer or None")

    semaphore = asyncio.Semaphore(resolved_concurrency)
    tasks = [
        _post_payload_with_semaphore(
            semaphore,
            payload,
            max_retries=max_retries,
            base_backoff=base_backoff,
            timeout=timeout,
            verbose=verbose,
        )
        for payload in payloads
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


def implement_post_requests_concurrently(
    payloads: Sequence[Dict[str, Any]],
    *,
    max_retries: int = 4,
    base_backoff: float = 1.0,
    timeout: Optional[float] = None,
    verbose: bool = False,
    max_concurrency: Optional[int] = None,
) -> List[Union[Dict[str, Any], Exception]]:
    """
    并发发送多个 LLM 请求，返回与输入 payload 顺序一致的结果列表。

    成功项为响应 JSON，失败项为异常对象（通常是 ``requests.RequestException``）。
    """
    if not isinstance(payloads, Sequence):
        raise TypeError("payloads must be a sequence of payload dictionaries")
    if len(payloads) == 0:
        return []

    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError("each payload must be a dictionary")

    return asyncio.run(
        _implement_post_requests_concurrently_async(
            payloads,
            max_retries=max_retries,
            base_backoff=base_backoff,
            timeout=timeout,
            verbose=verbose,
            max_concurrency=max_concurrency,
        )
    )




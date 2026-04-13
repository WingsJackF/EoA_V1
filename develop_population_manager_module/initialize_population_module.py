from __future__ import annotations

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: add_seed_individual (robust import/fallbacks)

Description:
    Adds the task-provided seed entry code and its
    associated strategic thought as the first seed individual, validates and
    evaluates it, and returns a formatted individual dictionary.

Improvements over previous version:
    - Avoids top-level imports that may raise ModuleNotFoundError during import time.
    - Attempts to import validator and evaluator at runtime and provides robust
      fallbacks if those modules are not available.
    - Catches and surfaces specific exceptions (ImportError, SyntaxError, ValueError, TypeError).
    - Conforms to constraints: no file I/O, does not modify evaluation logic.

Note:
    - The function tries to use existing project modules when available.
    - If the embedded evaluator module is missing, evaluation will fail with ImportError.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from tasks.base import EvolutionTask

from tasks.base import FAILED_COMBINED_SCORE


def _fallback_validate_constructor_syntax(code_str: str, function_name: str = "_entrypoint") -> None:
    """
    Local fallback validator for entry-function syntax and signature.

    Raises:
        SyntaxError: If the code is syntactically invalid.
        ValueError: If the required function is missing or has invalid signature.
        TypeError: If input is not a string.
    """
    import ast

    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    src = code_str.strip()
    if not src:
        raise ValueError("Provided code string is empty")

    try:
        module_ast = ast.parse(src)
    except SyntaxError as se:
        # Re-raise with readable message
        raise SyntaxError(f"Syntax error while parsing code: {se.msg} (line {se.lineno})") from se

    found = False
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            found = True
            func_def = node

            posonly_count = len(getattr(func_def.args, "posonlyargs", ()))
            pos_args_count = len(func_def.args.args)
            vararg = func_def.args.vararg
            kwonly_count = len(func_def.args.kwonlyargs)
            kw_defaults = func_def.args.kw_defaults
            kwarg = func_def.args.kwarg

            if posonly_count != 0 or pos_args_count != 0:
                raise ValueError(f"Function {function_name!r} must accept no positional arguments.")
            if vararg is not None:
                raise ValueError(f"Function {function_name!r} must not accept *args (varargs).")
            if kwonly_count != 0 and any(d is None for d in kw_defaults):
                raise ValueError(f"Function {function_name!r} must not have required keyword-only arguments.")
            if kwarg is not None:
                raise ValueError(f"Function {function_name!r} must not accept **kwargs.")
            # valid signature
            return None

    if not found:
        raise ValueError(f"Function {function_name!r} not found in provided code.")


def _fallback_evaluate_constructor_code(code_str: str) -> Dict[str, Any]:
    """
    Local wrapper that attempts to call run_evaluation from the embedded evaluator module.
    This is used if the project's define_evaluator_api_interface is not importable.

    Raises:
        ImportError: If the embedded evaluator module cannot be found.
    """
    # Import at runtime to allow flexible project layout; may raise ImportError.
    try:
        from integrate_fitness_evaluator_module.embed_builtin_evaluator_code import run_evaluation  # type: ignore
    except ImportError as imp_err:
        # Surface ImportError to caller with a clear message
        raise ImportError(
            "Embedded evaluator not available: cannot evaluate task code."
        ) from imp_err

    # Call the embedded evaluator (it returns a dict according to spec)
    return run_evaluation(code_str)


def add_seed_individual(task: EvolutionTask) -> Dict[str, Any]:
    """
    使用任务定义的种子代码与策略说明，校验并评估后返回种子个体。

    Raises:
        SyntaxError, ValueError: 校验失败。
    """
    from tasks.base import EvolutionTask as _ET

    if not isinstance(task, _ET):
        raise TypeError("task must be an EvolutionTask instance")

    seed_code = task.seed_code
    seed_thought = task.seed_thought
    eval_start = time.perf_counter()
    log_code_run_start(0, 0, phase="seed")

    try:
        task.validate_syntax(seed_code)
    except ImportError:
        _fallback_validate_constructor_syntax(seed_code, function_name=task.target_function_name)

    fitness_result = task.evaluate(seed_code)
    elapsed = time.perf_counter() - eval_start
    try:
        if float(fitness_result.get("eval_time", 0.0)) <= 0.0:
            fitness_result["eval_time"] = elapsed
    except (TypeError, ValueError, AttributeError):
        pass
    log_code_run_finish(
        0,
        0,
        success=not bool(fitness_result.get("error")) if isinstance(fitness_result, dict) else False,
        elapsed=elapsed,
        fitness=fitness_result if isinstance(fitness_result, dict) else None,
        error=None if isinstance(fitness_result, dict) else "Seed evaluator returned non-dict result",
        phase="seed",
    )

    return {
        "thought": seed_thought,
        "code": seed_code,
        "fitness": fitness_result,
    }



import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: generate_diverse_initial_individuals

Description:
    Generates diverse initial individuals by interacting with the LLM module,
    using the provided seed code and strategic thought as references.
    For each requested individual, this module:
      - Crafts a diversity-focused prompt,
      - Calls the LLM via the API wrapper,
      - Extracts thought and code from the LLM response,
      - Validates the task entry code,
      - Evaluates the candidate via the task-bound evaluator,
      - Formats and returns the individual with fitness metadata.

Notes:
    - No file I/O is performed.
    - The module assumes the project's LLM interaction, response parser, syntax
      validator, and evaluator modules are available at runtime.
    - Network/API errors and parsing/validation failures are captured per-individual
      and recorded in the returned individual's 'fitness' field as an 'error' entry.
    - This module avoids broad except clauses; specific exception types are handled.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import random
import requests  # used to reference requests.RequestException in exception handling
import time

from run_output_recorder import log_code_run_finish, log_code_run_start
from tasks.task_support.eval_timeout import eval_timeout_is_disabled, run_spawn_eval_jobs, spawn_initial_eval_worker
from tasks.task_support.processes import cleanup_process_pool


def _initial_failure_fitness() -> Dict[str, Any]:
    return {
        "combined_score": FAILED_COMBINED_SCORE,
        "eval_time": 0.0,
        "error": None,
    }


def _evaluate_initial_raw_content(task_id: str, raw_content: str, logical_cuda_device: int | None = None) -> Dict[str, Any]:
    from tasks.task_support.gpu import configure_logical_cuda_device
    from tasks import get_task

    configure_logical_cuda_device(logical_cuda_device)
    task = get_task(task_id)
    failure_fitness = _initial_failure_fitness()
    eval_start = time.perf_counter()

    try:
        sections = task.extract_thought_and_code(raw_content)
        thought_text = sections.get("thought", "").strip()
        code_text = sections.get("code", "").strip()
    except (ValueError, TypeError) as parse_err:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": "",
                "code": raw_content[:1000],
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"Parsing failed: {str(parse_err)}"},
            },
            "elapsed": elapsed,
        }

    try:
        task.validate_syntax(code_text)
    except SyntaxError as se:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": thought_text,
                "code": code_text,
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"SyntaxError: {str(se)}"},
            },
            "elapsed": elapsed,
        }
    except ValueError as ve:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": thought_text,
                "code": code_text,
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"Validation Error: {str(ve)}"},
            },
            "elapsed": elapsed,
        }
    except TypeError as te:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": thought_text,
                "code": code_text,
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"Validation TypeError: {str(te)}"},
            },
            "elapsed": elapsed,
        }

    try:
        fitness = task.evaluate(code_text)
        elapsed = time.perf_counter() - eval_start
        if not isinstance(fitness, dict):
            fitness = {**failure_fitness, "eval_time": elapsed, "error": "Evaluator returned non-dict result"}
        else:
            if "error" not in fitness:
                fitness["error"] = None
            try:
                if float(fitness.get("eval_time", 0.0)) <= 0.0:
                    fitness["eval_time"] = elapsed
            except (TypeError, ValueError):
                fitness["eval_time"] = elapsed
    except ImportError as ie:
        elapsed = time.perf_counter() - eval_start
        fitness = {**failure_fitness, "eval_time": elapsed, "error": f"Evaluator import failed: {str(ie)}"}
    except TypeError as te:
        elapsed = time.perf_counter() - eval_start
        fitness = {**failure_fitness, "eval_time": elapsed, "error": f"Evaluator TypeError: {str(te)}"}

    return {
        "individual": {
            "thought": thought_text,
            "code": code_text,
            "fitness": fitness,
        },
        "elapsed": elapsed,
    }


def _log_initial_result(response_index: int, result: Dict[str, Any]) -> None:
    individual = result["individual"]
    fitness = individual.get("fitness", {})
    log_code_run_finish(
        0,
        response_index,
        success=not bool(fitness.get("error")),
        elapsed=float(result.get("elapsed", 0.0)),
        fitness=fitness if isinstance(fitness, dict) else None,
        error=None if isinstance(fitness, dict) else "Worker returned invalid fitness payload",
        phase="initial",
    )


def _evaluate_initial_raw_contents(
    task: "EvolutionTask",
    raw_items: List[tuple[int, str]],
    *,
    evaluation_concurrency: int,
    evaluation_gpu_logical_ids: list[int] | None = None,
    evaluation_timeout_seconds: float | None = None,
) -> Dict[int, Dict[str, Any]]:
    if not raw_items:
        return {}

    results: Dict[int, Dict[str, Any]] = {}

    if not eval_timeout_is_disabled(evaluation_timeout_seconds):
        limit = float(evaluation_timeout_seconds)
        max_workers = min(evaluation_concurrency, len(raw_items))
        jobs: List[Tuple[int, Tuple[Any, ...]]] = []
        for response_index, raw_content in raw_items:
            log_code_run_start(0, response_index, phase="initial")
            logical_cuda_device = None
            if evaluation_gpu_logical_ids:
                logical_cuda_device = evaluation_gpu_logical_ids[
                    response_index % len(evaluation_gpu_logical_ids)
                ]
            jobs.append((response_index, (task.id, raw_content, logical_cuda_device)))
        results_map = run_spawn_eval_jobs(
            jobs,
            max_workers=max_workers,
            job_timeout_seconds=limit,
            worker_target=spawn_initial_eval_worker,
            on_job_complete=_log_initial_result,
        )
        for response_index, _raw in raw_items:
            results[response_index] = results_map[response_index]["individual"]
        return results

    if evaluation_concurrency == 1 or len(raw_items) == 1:
        for response_index, raw_content in raw_items:
            log_code_run_start(0, response_index, phase="initial")
            logical_cuda_device = None
            if evaluation_gpu_logical_ids:
                logical_cuda_device = evaluation_gpu_logical_ids[response_index % len(evaluation_gpu_logical_ids)]
            result = _evaluate_initial_raw_content(task.id, raw_content, logical_cuda_device)
            _log_initial_result(response_index, result)
            results[response_index] = result["individual"]
        return results

    max_workers = min(evaluation_concurrency, len(raw_items))
    spawn_context = mp.get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_index = {}
            for response_index, raw_content in raw_items:
                log_code_run_start(0, response_index, phase="initial")
                logical_cuda_device = None
                if evaluation_gpu_logical_ids:
                    logical_cuda_device = evaluation_gpu_logical_ids[response_index % len(evaluation_gpu_logical_ids)]
                future = executor.submit(_evaluate_initial_raw_content, task.id, raw_content, logical_cuda_device)
                future_to_index[future] = response_index
            for future in as_completed(future_to_index):
                response_index = future_to_index[future]
                result = future.result()
                _log_initial_result(response_index, result)
                results[response_index] = result["individual"]
    except KeyboardInterrupt:
        cleanup_process_pool(executor)
        raise

    return results


def generate_diverse_initial_individuals(
    task: EvolutionTask,
    n_individuals: int,
    *,
    max_concurrency: int = 8,
    evaluation_concurrency: int = 1,
    evaluation_gpu_logical_ids: list[int] | None = None,
    evaluation_timeout_seconds: float | None = None,
) -> List[Dict[str, Any]]:
    """
    基于任务的种子与提示配置，用 LLM 生成多样化初始个体。

    Returns:
        每个元素为 ``thought`` / ``code`` / ``fitness``；失败时在 fitness 中带 ``error``。
    """
    from tasks.base import EvolutionTask as _ET

    if not isinstance(task, _ET):
        raise TypeError("task must be an EvolutionTask instance")
    if not isinstance(n_individuals, int) or n_individuals < 0:
        raise ValueError("n_individuals must be a non-negative integer")
    if not isinstance(evaluation_concurrency, int) or evaluation_concurrency <= 0:
        raise ValueError("evaluation_concurrency must be a positive integer")

    diverse_individuals: List[Dict[str, Any]] = []

    try:
        from implement_llm_interaction_module.develop_api_wrapper import (  # type: ignore
            construct_api_request_payload,
            implement_post_requests_concurrently,
        )
        from implement_llm_interaction_module.llm_config import get_llm_settings  # type: ignore
    except ImportError as ie:
        raise ImportError(
            "LLM API wrapper module not found. Ensure implement_llm_interaction_module/develop_api_wrapper.py is available."
        ) from ie

    diversity_instructions = task.diversity_instructions
    system_prompt_template = task.initial_population_system_prompt

    payloads: List[Dict[str, Any]] = []
    for i in range(n_individuals):
        diversity_instruction = random.choice(diversity_instructions)

        user_prompt = task.format_diversity_user_prompt(diversity_instruction)

        # Construct payload and call LLM
        payload = construct_api_request_payload(
            system_prompt_template,
            user_prompt,
            model=get_llm_settings().model,
        )
        payloads.append(payload)

    response_results = implement_post_requests_concurrently(
        payloads,
        max_retries=3,
        base_backoff=10.0,
        timeout=180.0,
        verbose=True,
        max_concurrency=max_concurrency,
    )

    raw_items: List[tuple[int, str]] = []
    pending_individuals: List[Dict[str, Any] | None] = [None] * len(response_results)

    for response_index, response_result in enumerate(response_results):
        # Default fitness placeholder in case of failure during generation/eval
        failure_fitness = _initial_failure_fitness()

        if isinstance(response_result, Exception):
            # Record the failure and continue to next individual
            pending_individuals[response_index] = {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "error": f"LLM request failed: {str(response_result)}"}
            }
            continue

        response_json = response_result

        # Parse LLM response content
        try:
            # Navigate typical response structure
            raw_content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as key_err:
            log_code_run_start(0, response_index, phase="initial")
            log_code_run_finish(
                0,
                response_index,
                success=False,
                elapsed=0.0,
                error=f"Unexpected LLM response structure: {str(key_err)}",
                phase="initial",
            )
            pending_individuals[response_index] = {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "error": f"Unexpected LLM response structure: {str(key_err)}"}
            }
            continue

        raw_items.append((response_index, raw_content))

    evaluated_items = _evaluate_initial_raw_contents(
        task,
        raw_items,
        evaluation_concurrency=evaluation_concurrency,
        evaluation_gpu_logical_ids=evaluation_gpu_logical_ids,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
    )

    for response_index, individual in evaluated_items.items():
        pending_individuals[response_index] = individual

    diverse_individuals.extend(ind for ind in pending_individuals if ind is not None)

    return diverse_individuals

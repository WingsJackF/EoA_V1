import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: define_strategy_selection_policy

Description:
    Defines policy logic for selecting which evolutionary prompt strategy to apply
    at each offspring generation step. Supports fixed-ratio and simple adaptive
    policies responding to stagnation count and early-generation exploration boost.

Usage:
    Call define_strategy_selection_policy(context) with a context dict containing:
      - generation: int
      - recent_scores: list[float] (optional)
      - stagnation_count: int (optional)
      - default_ratios: dict (optional) with keys 'modification','exploration','simplification'
      - adaptive_strategy: bool (optional, default True)

Returns:
    (chosen_strategy: str, probabilities: Dict[str, float])
"""

from typing import Any, Dict, Optional, Tuple
import random


def _normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize ratio values to a valid probability distribution. Negative values are
    clipped to zero. If sum is zero after clipping, returns an even distribution.
    """
    # Clip negatives
    clipped = {k: (float(v) if isinstance(v, (int, float)) else 0.0) for k, v in ratios.items()}
    for k in list(clipped.keys()):
        if clipped[k] < 0.0:
            clipped[k] = 0.0

    total = sum(clipped.values())
    if total <= 0.0:
        # Fallback to uniform distribution
        n = len(clipped) if len(clipped) > 0 else 3
        uniform = 1.0 / n
        return {k: uniform for k in clipped} if len(clipped) > 0 else {
            "modification": 1.0 / 3,
            "exploration": 1.0 / 3,
            "simplification": 1.0 / 3
        }

    return {k: clipped[k] / total for k in clipped}


def define_strategy_selection_policy(
    context: Dict[str, Any]
) -> Tuple[str, Dict[str, float]]:
    """
    Determines which evolutionary strategy to apply at this generation step.

    Args:
        context: Dictionary containing generation context. Expected keys:
            - generation: int
            - recent_scores: list[float] (optional)
            - stagnation_count: int (optional)
            - default_ratios: Dict[str, float] (optional)
            - adaptive_strategy: bool (optional)

    Returns:
        (strategy, probabilities): Tuple of selected strategy string and the
        probability distribution used for selection.
    """
    # Default ratios if not provided
    default_ratios = {
        "modification": 0.5,
        "exploration": 0.3,
        "simplification": 0.2
    }

    # Extract context safely
    gen = int(context.get("generation", 0)) if isinstance(context.get("generation", 0), int) else 0
    stagnation = int(context.get("stagnation_count", 0)) if isinstance(context.get("stagnation_count", 0), int) else 0
    provided = context.get("default_ratios")
    if isinstance(provided, dict):
        # Merge provided with defaults to ensure all keys present
        ratios = {
            "modification": provided.get("modification", default_ratios["modification"]),
            "exploration": provided.get("exploration", default_ratios["exploration"]),
            "simplification": provided.get("simplification", default_ratios["simplification"]),
        }
    else:
        ratios = default_ratios.copy()

    adaptive_enabled = bool(context.get("adaptive_strategy", True))
    exploration_is_active = ratios.get("exploration", 0.0) > 0.0

    # Adaptive adjustments. Strategies explicitly set to zero stay disabled for ablations.
    if adaptive_enabled and exploration_is_active:
        # If stagnation is high, boost exploration.
        if stagnation >= 5:
            boost = 0.3
        # Early generation: encourage exploration.
        elif gen < 5:
            boost = 0.2
        else:
            boost = 0.0

        if boost > 0.0:
            ratios["exploration"] = ratios.get("exploration", 0.0) + boost
            reducible = [
                key for key in ("modification", "simplification")
                if ratios.get(key, 0.0) > 0.0
            ]
            if reducible:
                reduction = boost / len(reducible)
                for key in reducible:
                    ratios[key] = max(0.0, ratios.get(key, 0.0) - reduction)

    strategy_stats = context.get("strategy_stats")
    if adaptive_enabled and isinstance(strategy_stats, dict):
        for key in ("modification", "exploration", "simplification"):
            if ratios.get(key, 0.0) <= 0.0:
                continue
            record = strategy_stats.get(key, {})
            if not isinstance(record, dict):
                continue
            attempted = float(record.get("attempted", 0) or 0)
            if attempted < 3:
                continue
            valid = float(record.get("valid", 0) or 0)
            invalid = float(record.get("invalid", 0) or 0)
            parent_improvements = float(record.get("parent_improvements", 0) or 0)
            best_improvements = float(record.get("best_improvements", 0) or 0)

            valid_rate = valid / attempted if attempted > 0 else 0.0
            invalid_rate = invalid / attempted if attempted > 0 else 0.0
            parent_improve_rate = parent_improvements / valid if valid > 0 else 0.0
            best_rate = best_improvements / attempted if attempted > 0 else 0.0
            quality = (
                0.45 * parent_improve_rate
                + 0.35 * valid_rate
                + 0.20 * min(1.0, best_rate * 4.0)
                - 0.25 * invalid_rate
            )
            multiplier = max(0.50, min(1.80, 0.65 + quality))
            ratios[key] = ratios.get(key, 0.0) * multiplier
    # Else: keep provided/default ratios

    # Normalize to valid probabilities
    probabilities = _normalize_ratios(ratios)

    # Randomly choose according to probabilities
    strategies = list(probabilities.keys())
    probs = [probabilities[s] for s in strategies]
    # Ensure reproducible selection if context supplies a seed (optional)
    seed = context.get("random_seed")
    if isinstance(seed, int):
        rand = random.Random(seed)
        chosen = rand.choices(strategies, weights=probs, k=1)[0]
    else:
        chosen = random.choices(strategies, weights=probs, k=1)[0]

    return chosen, probabilities



import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: orchestrate_parent_selection_and_prompt_preparation

Description:
    Orchestrates parent selection from the current population according to a chosen
    evolutionary strategy and prepares the final API request payload for the LLM.
    The function supports three strategies:
      - "modification": select one fitness-biased parent and request a targeted improvement
      - "simplification": select one fitness-biased parent and request a simplification
      - "exploration": select two fitness-biased parents and request synthesis

    The function returns a payload dictionary ready to be sent to the LLM chat/completions
    endpoint via the project's API wrapper (construct_api_request_payload).

Notes:
    - This module does not perform any LLM calls or fitness evaluation itself.
    - Parent selection uses tournament pressure plus a small random branch for diversity.
    - The function expects prompt_strategies to be a mapping from strategy name to a
      callable that returns (system_prompt, user_prompt) when provided the appropriate
      parent(s).
"""

from typing import Any, Callable, Dict, List, Optional
import random

# Import the payload constructor from the project's LLM interaction module.
# This is required by the main function logic.
from implement_llm_interaction_module.develop_api_wrapper import construct_api_request_payload  # type: ignore
from implement_llm_interaction_module.llm_config import get_llm_settings  # type: ignore


def _combined_score_of(individual: Dict[str, Any]) -> float:
    fitness = individual.get("fitness", {})
    if not isinstance(fitness, dict):
        return float("-inf")
    try:
        return float(fitness.get("combined_score", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _parent_summary(parent: Dict[str, Any]) -> Dict[str, Any]:
    code = str(parent.get("code", ""))
    thought = str(parent.get("thought", ""))
    return {
        "combined_score": _combined_score_of(parent),
        "thought_preview": thought[:120],
        "code_length": len(code),
    }


def _select_parent_by_fitness(
    population: List[Dict[str, Any]],
    *,
    tournament_size: int = 3,
    top_fraction: float = 0.6,
    random_parent_probability: float = 0.15,
) -> Dict[str, Any]:
    if not population:
        raise ValueError("population is empty; cannot select parents")
    if random.random() < random_parent_probability:
        return random.choice(population)

    ranked = sorted(population, key=_combined_score_of, reverse=True)
    top_count = max(1, int(len(ranked) * top_fraction))
    top_count = min(len(ranked), max(top_count, tournament_size))
    candidate_pool = ranked[:top_count]
    k = min(max(1, tournament_size), len(candidate_pool))
    contestants = random.sample(candidate_pool, k=k)
    return max(contestants, key=_combined_score_of)


def _select_parents_by_fitness(
    population: List[Dict[str, Any]],
    count: int,
    *,
    tournament_size: int = 3,
    top_fraction: float = 0.6,
    random_parent_probability: float = 0.15,
) -> List[Dict[str, Any]]:
    remaining = list(population)
    selected: List[Dict[str, Any]] = []
    for _ in range(min(count, len(remaining))):
        parent = _select_parent_by_fitness(
            remaining,
            tournament_size=tournament_size,
            top_fraction=top_fraction,
            random_parent_probability=random_parent_probability,
        )
        selected.append(parent)
        for idx, item in enumerate(remaining):
            if item is parent:
                del remaining[idx]
                break
    return selected


def orchestrate_parent_selection_and_prompt_preparation(
    population: List[Dict[str, Any]],
    strategy: str,
    prompt_strategies: Dict[str, Callable[..., tuple]],
    model: Optional[str] = None,
    parent_selection_policy: Optional[Dict[str, Any]] = None,
    return_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Selects parent(s) according to strategy, prepares prompts, and constructs API payload.

    Args:
        population: List of individual dictionaries. Each individual should have at least
                    'thought' and 'code' fields. Fitness is not required for selection here.
        strategy: Strategy identifier ("modification", "exploration", "simplification").
        prompt_strategies: Mapping from strategy name to prompt strategy function.
                           - For 'modification' and 'simplification', the function should
                             accept a single parent dict and return (system_prompt, user_prompt).
                           - For 'exploration', the function should accept a list of parent dicts.
        model: 模型名；默认 ``get_llm_settings().model``。
        parent_selection_policy: 父代选择参数。默认使用 fitness-aware tournament。
        return_metadata: 为 True 时返回 {"payload": ..., "selection": ...}。

    Returns:
        API request payload (dict) ready for LLM interaction. Uses construct_api_request_payload.

    Raises:
        ValueError: If the population is empty or the strategy is unknown.
        TypeError: If prompt_strategies does not contain the required callable.
    """
    # Validate population
    if not isinstance(population, list):
        raise TypeError("population must be a list of individuals")
    if len(population) == 0:
        raise ValueError("population is empty; cannot select parents")

    # Validate strategy mapping
    if not isinstance(prompt_strategies, dict):
        raise TypeError("prompt_strategies must be a dict mapping strategy names to callables")
    if strategy not in prompt_strategies:
        raise ValueError(f"Unknown strategy: {strategy}")

    prompt_fn = prompt_strategies[strategy]
    if not callable(prompt_fn):
        raise TypeError(f"Prompt strategy for '{strategy}' is not callable")

    selection_policy = parent_selection_policy or {}
    tournament_size = int(selection_policy.get("tournament_size", 3))
    top_fraction = float(selection_policy.get("top_fraction", 0.6))
    random_parent_probability = float(selection_policy.get("random_parent_probability", 0.15))

    selected_parents: List[Dict[str, Any]] = []

    # Select parents according to strategy
    if strategy in ("modification", "simplification"):
        parent = _select_parent_by_fitness(
            population,
            tournament_size=tournament_size,
            top_fraction=top_fraction,
            random_parent_probability=random_parent_probability,
        )
        selected_parents = [parent]
        # Call the prompt function with the single parent
        system_prompt, user_prompt = prompt_fn(parent)
    elif strategy == "exploration":
        # Exploration: select two parents if possible, else select as many as available (at least 2 was required).
        k = 2 if len(population) >= 2 else len(population)
        parents = _select_parents_by_fitness(
            population,
            k,
            tournament_size=tournament_size,
            top_fraction=top_fraction,
            random_parent_probability=random_parent_probability,
        )
        selected_parents = parents
        system_prompt, user_prompt = prompt_fn(parents)
    else:
        # Defensive: should not happen given earlier check, but keep consistent error
        raise ValueError(f"Unhandled strategy: {strategy}")

    resolved_model = model if model is not None else get_llm_settings().model
    payload = construct_api_request_payload(system_prompt, user_prompt, model=resolved_model)
    if return_metadata:
        parent_scores = [_combined_score_of(parent) for parent in selected_parents]
        return {
            "payload": payload,
            "selection": {
                "strategy": strategy,
                "parent_scores": parent_scores,
                "parent_best_score": max(parent_scores) if parent_scores else None,
                "parents": [_parent_summary(parent) for parent in selected_parents],
                "parent_selection_policy": {
                    "mode": "fitness_tournament",
                    "tournament_size": tournament_size,
                    "top_fraction": top_fraction,
                    "random_parent_probability": random_parent_probability,
                },
            },
        }
    return payload



import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: collect_and_integrate_offspring_results

Description:
    Processes raw LLM offspring content strings, extracts strategic thought and
    candidate code, validates syntax, evaluates using the task-bound evaluator,
    and returns a list of evaluated individuals ready for population integration.

Constraints:
    - Python 3.9+
    - No file I/O.
    - Uses project parser, validator, and evaluator functions at runtime.
    - Does not swallow broad exceptions; handles specific exception types.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import time

from run_output_recorder import log_code_run_finish, log_code_run_start
from tasks.base import FAILED_COMBINED_SCORE
from tasks.task_support.eval_timeout import eval_timeout_is_disabled, run_spawn_eval_jobs, spawn_offspring_eval_worker
from tasks.task_support.processes import cleanup_process_pool

if TYPE_CHECKING:
    from tasks.base import EvolutionTask


def _offspring_failure_fitness() -> Dict[str, Any]:
    return {
        "combined_score": FAILED_COMBINED_SCORE,
        "eval_time": 0.0,
        "error": None,
    }


def _evaluate_single_offspring(task_id: str, raw_content: Any, logical_cuda_device: int | None = None) -> Dict[str, Any]:
    from tasks.task_support.gpu import configure_logical_cuda_device
    from tasks import get_task

    configure_logical_cuda_device(logical_cuda_device)
    task = get_task(task_id)
    failure_fitness = _offspring_failure_fitness()

    if not isinstance(raw_content, str):
        return {
            "individual": {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "error": "Raw offspring content is not a string"},
            },
            "elapsed": 0.0,
        }

    eval_start = time.perf_counter()

    try:
        sections = task.extract_thought_and_code(raw_content)
        thought = sections.get("thought", "").strip()
        code = sections.get("code", "").strip()
    except ValueError as ve:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"Extraction failed: {str(ve)}"},
            },
            "elapsed": elapsed,
        }
    except TypeError as te:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"Extraction TypeError: {str(te)}"},
            },
            "elapsed": elapsed,
        }

    try:
        task.validate_syntax(code)
    except SyntaxError as se:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": thought,
                "code": code,
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"SyntaxError: {str(se)}"},
            },
            "elapsed": elapsed,
        }
    except ValueError as ve:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": thought,
                "code": code,
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"Validation Error: {str(ve)}"},
            },
            "elapsed": elapsed,
        }
    except TypeError as te:
        elapsed = time.perf_counter() - eval_start
        return {
            "individual": {
                "thought": thought,
                "code": code,
                "fitness": {**failure_fitness, "eval_time": elapsed, "error": f"Validation TypeError: {str(te)}"},
            },
            "elapsed": elapsed,
        }

    try:
        fitness = task.evaluate(code)
        elapsed = time.perf_counter() - eval_start
        if not isinstance(fitness, dict):
            fitness = {
                **failure_fitness,
                "eval_time": elapsed,
                "error": "Evaluator returned non-dict result",
            }
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
        fitness = {**failure_fitness, "eval_time": elapsed, "error": f"Evaluator import error: {str(ie)}"}
    except TypeError as te:
        elapsed = time.perf_counter() - eval_start
        fitness = {**failure_fitness, "eval_time": elapsed, "error": f"Evaluator TypeError: {str(te)}"}

    return {
        "individual": {
            "thought": thought,
            "code": code,
            "fitness": fitness,
        },
        "elapsed": elapsed,
    }


def _log_offspring_result(iteration: int, code_index: int, result: Dict[str, Any]) -> None:
    individual = result["individual"]
    fitness = individual.get("fitness", {})
    log_code_run_finish(
        iteration,
        code_index,
        success=not bool(fitness.get("error")),
        elapsed=float(result.get("elapsed", 0.0)),
        fitness=fitness if isinstance(fitness, dict) else None,
        error=None if isinstance(fitness, dict) else "Worker returned invalid fitness payload",
        phase="offspring",
    )


def collect_and_integrate_offspring_results(
    raw_offspring_contents: List[str],
    task: "EvolutionTask",
    *,
    iteration: int = 0,
    code_index_start: int = 0,
    evaluation_concurrency: int = 1,
    evaluation_gpu_logical_ids: Optional[List[int]] = None,
    evaluation_timeout_seconds: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Processes LLM-generated offspring, evaluates, and formats them.

    Args:
        raw_offspring_contents: List of raw LLM response content strings (each containing thought and code).
        task: 当前演化任务（解析、校验、评估均绑定任务）。

    Returns:
        List of evaluated individuals, each as a dict with 'thought', 'code', and 'fitness'.
        If extraction/validation/evaluation fails for an item, its 'fitness' will contain an 'error' entry
        describing the failure and combined_score set to FAILED_COMBINED_SCORE.
    """
    from tasks.base import EvolutionTask as _ET

    if not isinstance(task, _ET):
        raise TypeError("task must be an EvolutionTask instance")
    if not isinstance(raw_offspring_contents, list):
        raise TypeError("raw_offspring_contents must be a list of strings")
    if not isinstance(evaluation_concurrency, int) or evaluation_concurrency <= 0:
        raise ValueError("evaluation_concurrency must be a positive integer")
    if evaluation_gpu_logical_ids is not None and not isinstance(evaluation_gpu_logical_ids, list):
        raise TypeError("evaluation_gpu_logical_ids must be a list of ints or None")

    if not raw_offspring_contents:
        return []

    if not eval_timeout_is_disabled(evaluation_timeout_seconds):
        limit = float(evaluation_timeout_seconds)
        max_workers = min(evaluation_concurrency, len(raw_offspring_contents))
        jobs: List[Tuple[int, Tuple[Any, ...]]] = []
        for idx, raw_content in enumerate(raw_offspring_contents):
            code_index = code_index_start + idx
            log_code_run_start(iteration, code_index, phase="offspring")
            logical_cuda_device = None
            if evaluation_gpu_logical_ids:
                logical_cuda_device = evaluation_gpu_logical_ids[idx % len(evaluation_gpu_logical_ids)]
            jobs.append((idx, (task.id, raw_content, logical_cuda_device)))

        def _on_offspring_done(idx: int, result: Dict[str, Any]) -> None:
            _log_offspring_result(iteration, code_index_start + idx, result)

        results_map = run_spawn_eval_jobs(
            jobs,
            max_workers=max_workers,
            job_timeout_seconds=limit,
            worker_target=spawn_offspring_eval_worker,
            on_job_complete=_on_offspring_done,
        )
        return [results_map[idx]["individual"] for idx in range(len(raw_offspring_contents))]

    if evaluation_concurrency == 1 or len(raw_offspring_contents) == 1:
        evaluated_individuals: List[Dict[str, Any]] = []
        for idx, raw_content in enumerate(raw_offspring_contents):
            code_index = code_index_start + idx
            log_code_run_start(iteration, code_index, phase="offspring")
            logical_cuda_device = None
            if evaluation_gpu_logical_ids:
                logical_cuda_device = evaluation_gpu_logical_ids[idx % len(evaluation_gpu_logical_ids)]
            result = _evaluate_single_offspring(task.id, raw_content, logical_cuda_device)
            _log_offspring_result(iteration, code_index, result)
            evaluated_individuals.append(result["individual"])
        return evaluated_individuals

    results_by_index: Dict[int, Dict[str, Any]] = {}
    max_workers = min(evaluation_concurrency, len(raw_offspring_contents))
    spawn_context = mp.get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_index = {}
            for idx, raw_content in enumerate(raw_offspring_contents):
                code_index = code_index_start + idx
                log_code_run_start(iteration, code_index, phase="offspring")
                logical_cuda_device = None
                if evaluation_gpu_logical_ids:
                    logical_cuda_device = evaluation_gpu_logical_ids[idx % len(evaluation_gpu_logical_ids)]
                future = executor.submit(_evaluate_single_offspring, task.id, raw_content, logical_cuda_device)
                future_to_index[future] = idx
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                code_index = code_index_start + idx
                result = future.result()
                _log_offspring_result(iteration, code_index, result)
                results_by_index[idx] = result["individual"]
    except KeyboardInterrupt:
        cleanup_process_pool(executor)
        raise

    return [results_by_index[idx] for idx in range(len(raw_offspring_contents))]




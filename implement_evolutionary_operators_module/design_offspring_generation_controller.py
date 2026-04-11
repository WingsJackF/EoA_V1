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

    # Adaptive adjustments
    # If stagnation is high, boost exploration
    if stagnation >= 5:
        # Increase exploration moderately, reduce others proportionally
        boost = 0.3
        ratios["exploration"] = ratios.get("exploration", 0.0) + boost
        # Reduce others proportionally (avoid going negative)
        reduction = boost / 2.0
        ratios["modification"] = max(0.0, ratios.get("modification", 0.0) - reduction)
        ratios["simplification"] = max(0.0, ratios.get("simplification", 0.0) - reduction)
    # Early generation: encourage exploration
    elif gen < 5:
        boost = 0.2
        ratios["exploration"] = ratios.get("exploration", 0.0) + boost
        reduction = boost / 2.0
        ratios["modification"] = max(0.0, ratios.get("modification", 0.0) - reduction)
        ratios["simplification"] = max(0.0, ratios.get("simplification", 0.0) - reduction)
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
      - "modification": select one parent randomly and request a targeted improvement
      - "simplification": select one parent randomly and request a simplification
      - "exploration": select two parents (or more) randomly and request synthesis

    The function returns a payload dictionary ready to be sent to the LLM chat/completions
    endpoint via the project's API wrapper (construct_api_request_payload).

Notes:
    - This module does not perform any LLM calls or fitness evaluation itself.
    - Parent selection uses Python's random module to ensure diversity.
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


def orchestrate_parent_selection_and_prompt_preparation(
    population: List[Dict[str, Any]],
    strategy: str,
    prompt_strategies: Dict[str, Callable[..., tuple]],
    model: Optional[str] = None,
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

    # Select parents according to strategy
    if strategy in ("modification", "simplification"):
        # Single-parent strategies: select one parent randomly
        parent = random.choice(population)
        # Call the prompt function with the single parent
        system_prompt, user_prompt = prompt_fn(parent)
    elif strategy == "exploration":
        # Exploration: select two parents if possible, else select as many as available (at least 2 was required).
        k = 2 if len(population) >= 2 else len(population)
        parents = random.sample(population, k=k)
        system_prompt, user_prompt = prompt_fn(parents)
    else:
        # Defensive: should not happen given earlier check, but keep consistent error
        raise ValueError(f"Unhandled strategy: {strategy}")

    resolved_model = model if model is not None else get_llm_settings().model
    payload = construct_api_request_payload(system_prompt, user_prompt, model=resolved_model)
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

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from tasks.base import EvolutionTask


def collect_and_integrate_offspring_results(
    raw_offspring_contents: List[str], task: "EvolutionTask"
) -> List[Dict[str, Any]]:
    """
    Processes LLM-generated offspring, evaluates, and formats them.

    Args:
        raw_offspring_contents: List of raw LLM response content strings (each containing thought and code).
        task: 当前演化任务（解析、校验、评估均绑定任务）。

    Returns:
        List of evaluated individuals, each as a dict with 'thought', 'code', and 'fitness'.
        If extraction/validation/evaluation fails for an item, its 'fitness' will contain an 'error' entry
        describing the failure and combined_score set to 0.0.
    """
    from tasks.base import EvolutionTask as _ET

    if not isinstance(task, _ET):
        raise TypeError("task must be an EvolutionTask instance")
    if not isinstance(raw_offspring_contents, list):
        raise TypeError("raw_offspring_contents must be a list of strings")

    evaluated_individuals: List[Dict[str, Any]] = []

    for idx, raw_content in enumerate(raw_offspring_contents):
        # Prepare a default failure fitness template
        failure_fitness = {
            "min_max_ratio": 0.0,
            "combined_score": 0.0,
            "eval_time": 0.0,
            "error": None
        }

        if not isinstance(raw_content, str):
            individual = {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "error": "Raw offspring content is not a string"}
            }
            evaluated_individuals.append(individual)
            continue

        # 1) Extract thought and code
        try:
            sections = task.extract_thought_and_code(raw_content)
            thought = sections.get("thought", "").strip()
            code = sections.get("code", "").strip()
        except ValueError as ve:
            individual = {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "error": f"Extraction failed: {str(ve)}"}
            }
            evaluated_individuals.append(individual)
            continue
        except TypeError as te:
            individual = {
                "thought": "",
                "code": "",
                "fitness": {**failure_fitness, "error": f"Extraction TypeError: {str(te)}"}
            }
            evaluated_individuals.append(individual)
            continue

        # 2) Validate code syntax and required function presence
        try:
            task.validate_syntax(code)
        except SyntaxError as se:
            individual = {
                "thought": thought,
                "code": code,
                "fitness": {**failure_fitness, "error": f"SyntaxError: {str(se)}"}
            }
            evaluated_individuals.append(individual)
            continue
        except ValueError as ve:
            individual = {
                "thought": thought,
                "code": code,
                "fitness": {**failure_fitness, "error": f"Validation Error: {str(ve)}"}
            }
            evaluated_individuals.append(individual)
            continue
        except TypeError as te:
            individual = {
                "thought": thought,
                "code": code,
                "fitness": {**failure_fitness, "error": f"Validation TypeError: {str(te)}"}
            }
            evaluated_individuals.append(individual)
            continue

        # 3) Evaluate the code using task-bound evaluator
        try:
            fitness = task.evaluate(code)
            if not isinstance(fitness, dict):
                fitness = {**failure_fitness, "error": "Evaluator returned non-dict result"}
            else:
                if "error" not in fitness:
                    fitness["error"] = None
        except ImportError as ie:
            fitness = {**failure_fitness, "error": f"Evaluator import error: {str(ie)}"}
        except TypeError as te:
            fitness = {**failure_fitness, "error": f"Evaluator TypeError: {str(te)}"}
        # Note: Do not catch arbitrary Exception per module constraints.

        individual = {
            "thought": thought,
            "code": code,
            "fitness": fitness
        }
        evaluated_individuals.append(individual)

    return evaluated_individuals




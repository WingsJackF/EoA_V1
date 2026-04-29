import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
sys.path.append(project_root)


def _extract_gpu_args(argv: list[str]) -> list[int] | None:
    for idx, arg in enumerate(argv):
        if arg == "--gpu":
            if idx + 1 >= len(argv):
                return None
            try:
                return [int(argv[idx + 1])]
            except ValueError:
                return None
        if arg == "--gpus":
            if idx + 1 >= len(argv):
                return None
            raw = argv[idx + 1]
            try:
                return [int(part.strip()) for part in raw.split(",") if part.strip()]
            except ValueError:
                return None
    return None


_early_gpu_args = _extract_gpu_args(sys.argv[1:])
if _early_gpu_args:
    os.environ["EOA_USE_CUDA"] = "1"
    _visible = ",".join(str(gpu_id) for gpu_id in _early_gpu_args)
    os.environ["EOA_VISIBLE_GPU"] = _visible
    os.environ["EOA_CUDA_DEVICE_NUM"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible
r"""
Main executable for the task-pluggable evolutionary LLM+GP system.

Note:
- This script assembles previously implemented modules in the project tree.
- It performs real LLM calls using the configured API wrapper and performs
  real fitness evaluations via the task-bound evaluator. Ensure network access
  and scipy/numpy are installed in the runtime environment.
- Adjust parameters near the top of the file as desired.
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is importable (adjust path to the provided project root)
# PROJECT_ROOT = r"C:\Users\17640\Desktop\tree_cot\init_output_files\mini_list_2025-11-15_18-28-58"
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# Import project modules (these are the previously implemented components)
from develop_population_manager_module.initialize_population_module import (
    add_seed_individual,
    generate_diverse_initial_individuals,
)
from tasks import get_task, list_tasks
from tasks.base import EvolutionTask, individual_is_valid
from develop_population_manager_module.archive_best_individuals_module import archive_best_individuals
from develop_population_manager_module.select_parents_and_offspring_module import (
    implement_elitism_selection,
    implement_tournament_selection,
)
from implement_evolutionary_operators_module.design_offspring_generation_controller import (
    define_strategy_selection_policy,
    orchestrate_parent_selection_and_prompt_preparation,
    collect_and_integrate_offspring_results,
)
from implement_llm_interaction_module.develop_api_wrapper import (
    implement_post_requests_concurrently,
    safe_print,
)
from implement_llm_interaction_module.llm_config import configure_llm, get_llm_settings
from tasks.task_support.gpu import configure_gpu_environment, gpu_requested, visible_gpu, visible_gpu_ids
from tasks.task_support.eval_timeout import eval_timeout_is_disabled
from tasks.task_support.processes import cleanup_process_pool
from run_output_recorder import (
    make_run_output_dir,
    save_code_artifact,
    save_final_archive,
    save_named_result,
    save_final_test_result,
    save_generation_snapshot,
    tee_terminal_to_file,
    write_run_meta,
)

# -------------------- Configuration --------------------
POPULATION_SIZE = 32            # total individuals in population (including seed)
GENERATIONS = 10           # number of evolutionary generations
ELITISM_COUNT = 8              # number of elites to carry unchanged each generation
OFFSPRING_PER_GEN = 16          # number of offspring to generate per generation
DEFAULT_STRATEGY_RATIOS = {
    "modification": 0.4,
    "exploration": 0.4,
    "simplification": 0.2,
}
DEFAULT_LLM_CONCURRENCY = 8
DEFAULT_LLM_POST_TIMEOUT = 180.0
DEFAULT_EVAL_CONCURRENCY = 2
DEFAULT_EVAL_TIMEOUT_SECONDS = 300.0
# LLM：见 implement_llm_interaction_module/llm_config.py（环境变量 / --llm-* / LLM_CONFIG_FILE）
# -------------------------------------------------------


def _log_llm_settings() -> None:
    s = get_llm_settings()
    key = s.api_key
    if not key:
        masked = "(empty)"
    elif len(key) <= 8:
        masked = "***"
    else:
        masked = key[:4] + "…" + key[-2:]
    safe_print(
        f"LLM [{s.provider_label}] base_url={s.base_url} model={s.model} "
        f"timeout={s.timeout}s api_key={masked}"
    )


def _log_runtime_settings() -> None:
    if gpu_requested():
        gpu_ids = visible_gpu_ids()
        if len(gpu_ids) <= 1:
            safe_print(f"GPU selection: physical GPU {visible_gpu()} (mapped to cuda:0)")
        else:
            safe_print(
                "GPU selection: physical GPUs "
                f"{visible_gpu()} (workers map to logical cuda:{'/'.join(str(i) for i in range(len(gpu_ids)))})"
            )
    else:
        safe_print("GPU selection: CPU/default device")


def _summarize_individual(ind: Dict[str, Any]) -> str:
    """Return a concise string summarizing an individual for logging."""
    cs = None
    try:
        cs = ind.get("fitness", {}).get("combined_score")
    except Exception:
        cs = None
    thought = (ind.get("thought") or "")[:120]
    return f"combined_score={cs}, thought={thought}"


def _format_full_test_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return str(metrics)
    preferred_keys = (
        "combined_score",
        "objective",
        "teacher_score",
        "student_score",
        "gap_percent",
        "average_reward",
        "avg_bins",
        "l1_bound",
        "excess_percent",
        "min_max_ratio",
    )
    parts: List[str] = []
    seen = set()
    for key in preferred_keys:
        if key in metrics:
            parts.append(f"{key}={metrics.get(key)}")
            seen.add(key)
    for key, value in metrics.items():
        if key in seen:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def main(
    task: EvolutionTask,
    *,
    output_run_dir: Optional[Path] = None,
    llm_concurrency: int = DEFAULT_LLM_CONCURRENCY,
    eval_concurrency: int = DEFAULT_EVAL_CONCURRENCY,
    eval_gpu_logical_ids: Optional[List[int]] = None,
    eval_timeout_seconds: Optional[float] = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Main evolutionary loop orchestration.
    """
    # Initialize population with seed + LLM-generated diverse individuals
    if output_run_dir is not None:
        safe_print(f"Run output directory: {output_run_dir.resolve()}")
    safe_print(f"Task: {task.id} (entry: {task.target_function_name})")
    _log_llm_settings()
    _log_runtime_settings()
    if not eval_timeout_is_disabled(eval_timeout_seconds):
        safe_print(f"Per-individual eval wall-clock timeout: {float(eval_timeout_seconds):g} s")
    if llm_concurrency <= 0:
        raise ValueError("llm_concurrency must be a positive integer")
    if eval_concurrency <= 0:
        raise ValueError("eval_concurrency must be a positive integer")
    safe_print(
        "Initializing population with seed individual "
        f"(llm_concurrency={llm_concurrency}, eval_concurrency={eval_concurrency})..."
    )
    try:
        seed_ind = add_seed_individual(task)
    except (SyntaxError, ValueError, TypeError) as e:
        safe_print(f"Seed validation/evaluation failed: {e}")
        return []
    if not individual_is_valid(seed_ind):
        safe_print(f"Seed evaluation returned an error; aborting run: {seed_ind.get('fitness', {}).get('error')}")
        return []

    population: List[Dict[str, Any]] = [seed_ind]
    # Generate remaining individuals using the LLM interaction module
    n_to_generate = POPULATION_SIZE - 1
    if n_to_generate > 0:
        try:
            generated = generate_diverse_initial_individuals(
                task,
                n_to_generate,
                max_concurrency=llm_concurrency,
                evaluation_concurrency=eval_concurrency,
                evaluation_gpu_logical_ids=eval_gpu_logical_ids,
                evaluation_timeout_seconds=eval_timeout_seconds,
            )
            # Only append properly formatted dicts
            skipped_generated = 0
            for ind in generated:
                if individual_is_valid(ind):
                    population.append(ind)
                else:
                    skipped_generated += 1
            if skipped_generated:
                safe_print(f"Skipped {skipped_generated} invalid generated individuals during initialization.")
            fallback_offset = 0
            max_fallback_attempts = max(POPULATION_SIZE, n_to_generate * 2)
            while len(population) < POPULATION_SIZE and fallback_offset < max_fallback_attempts:
                try:
                    fallback = task.build_fallback_individual(fallback_offset)
                except (TypeError, ImportError, ValueError, SyntaxError) as eval_err:
                    safe_print(f"Fallback generation failed #{fallback_offset + 1}: {eval_err}")
                    break
                if individual_is_valid(fallback):
                    population.append(fallback)
                else:
                    safe_print(
                        f"Skipped invalid fallback individual #{fallback_offset + 1}: "
                        f"{fallback.get('fitness', {}).get('error')}"
                    )
                fallback_offset += 1
            if len(population) < POPULATION_SIZE:
                safe_print(
                    f"Initialization ended with {len(population)} valid individuals "
                    f"(target={POPULATION_SIZE})."
                )
        except ImportError as imp_err:
            safe_print(f"LLM generation module import failed: {imp_err}")
            # Fall back: let the task provide local fallback individuals.
            for s in range(n_to_generate):
                try:
                    fallback = task.build_fallback_individual(s)
                    if individual_is_valid(fallback):
                        population.append(fallback)
                    else:
                        safe_print(
                            f"Skipped invalid fallback individual #{s + 1}: "
                            f"{fallback.get('fitness', {}).get('error')}"
                        )
                except (TypeError, ImportError, ValueError, SyntaxError) as eval_err:
                    safe_print(f"Fallback generation failed #{s + 1}: {eval_err}")

    safe_print(f"Initial population size: {len(population)}")
    for idx, ind in enumerate(population, start=1):
        safe_print(f"  Individual {idx}: {_summarize_individual(ind)}")

    if output_run_dir is not None:
        save_generation_snapshot(
            output_run_dir,
            generation=0,
            label="initial",
            population=population,
            archive=[population[0]],
            extra={"POPULATION_SIZE": POPULATION_SIZE, "GENERATIONS": GENERATIONS},
        )

    # Initialize archive with seed individual
    archive: List[Dict[str, Any]] = [population[0]]

    # Track progress for strategy adaptation
    best_history: List[float] = []
    stagnation_count = 0
    best_combined_score = population[0].get("fitness", {}).get("combined_score", 0.0)

    prompt_strategies = task.prompt_strategies

    for gen in range(1, GENERATIONS + 1):
        safe_print(f"\n=== Generation {gen} ===")
        # Prepare context for strategy selection
        context = {
            "generation": gen,
            "recent_scores": best_history[-5:],
            "stagnation_count": stagnation_count,
            "default_ratios": DEFAULT_STRATEGY_RATIOS,
            # optional deterministic sampling seed is omitted for stochasticity
        }

        # Decide strategies for each offspring to produce
        offspring_raw_contents: List[str] = []
        strategies_used: List[str] = []
        prepared_offspring_requests: List[tuple[int, Dict[str, Any]]] = []
        for off_idx in range(OFFSPRING_PER_GEN):
            try:
                chosen_strategy, probs = define_strategy_selection_policy(context)
            except Exception as e:
                # Catch-all here is only for the outer control flow; use specific exception types where possible.
                safe_print(f"Strategy selection failed: {e}")
                chosen_strategy = "modification"  # fallback
            strategies_used.append(chosen_strategy)

            # Orchestrate parent selection and prompt preparation
            try:
                payload = orchestrate_parent_selection_and_prompt_preparation(
                    population, chosen_strategy, prompt_strategies
                )
            except (ValueError, TypeError) as e:
                safe_print(f"Parent selection / prompt preparation failed: {e}")
                continue
            prepared_offspring_requests.append((off_idx, payload))

        safe_print(
            f"Dispatching {len(prepared_offspring_requests)} LLM requests for generation {gen} "
            f"(llm_concurrency={llm_concurrency})..."
        )
        request_results = implement_post_requests_concurrently(
            [payload for _, payload in prepared_offspring_requests],
            max_retries=5,
            base_backoff=2.0,
            timeout=DEFAULT_LLM_POST_TIMEOUT,
            verbose=True,
            max_concurrency=llm_concurrency,
        )

        for (off_idx, _payload), response_result in zip(prepared_offspring_requests, request_results):
            if isinstance(response_result, Exception):
                safe_print(f"LLM request failed for offspring {off_idx+1} in gen {gen}: {response_result}")
                continue

            # Extract raw content of LLM message
            try:
                raw_content = response_result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                safe_print(f"Unexpected LLM response structure for offspring {off_idx+1} in gen {gen}: {e}")
                continue

            offspring_raw_contents.append(raw_content)

        safe_print(f"Generated {len(offspring_raw_contents)} offspring (strategies: {strategies_used})")
        safe_print(
            f"Evaluating {len(offspring_raw_contents)} offspring with process pool "
            f"(eval_concurrency={eval_concurrency})..."
        )

        # Integrate offspring: parse, validate, evaluate
        try:
            evaluated_offspring = collect_and_integrate_offspring_results(
                offspring_raw_contents,
                task,
                iteration=gen,
                code_index_start=0,
                evaluation_concurrency=eval_concurrency,
                evaluation_gpu_logical_ids=eval_gpu_logical_ids,
                evaluation_timeout_seconds=eval_timeout_seconds,
            )
        except ImportError as ie:
            safe_print(f"Integration failed due to missing modules: {ie}")
            evaluated_offspring = []
        except TypeError as te:
            safe_print(f"Integration type error: {te}")
            evaluated_offspring = []

        valid_offspring = [off for off in evaluated_offspring if individual_is_valid(off)]
        invalid_offspring_count = len(evaluated_offspring) - len(valid_offspring)
        if invalid_offspring_count:
            safe_print(f"Skipped {invalid_offspring_count} invalid offspring in generation {gen}.")

        safe_print(f"Evaluated offspring count: {len(valid_offspring)}")
        for idx, off in enumerate(valid_offspring, start=1):
            safe_print(f"  Offspring {idx}: {_summarize_individual(off)}")

        # Combine population and offspring for selection
        combined_population = population + valid_offspring

        # Apply elitism: keep top ELITISM_COUNT
        try:
            elites = implement_elitism_selection(combined_population, ELITISM_COUNT)
        except (TypeError, ValueError, KeyError) as e:
            safe_print(f"Elitism selection failed: {e}")
            elites = []

        # Fill remaining population slots via tournament selection
        remaining_slots = POPULATION_SIZE - len(elites)
        if remaining_slots < 0:
            remaining_slots = 0

        try:
            # Prepare the pool for tournaments (exclude elites to avoid duplicates)
            pool_for_tournament = [ind for ind in combined_population if ind not in elites]
            tournament_selected = implement_tournament_selection(pool_for_tournament, remaining_slots, tournament_size=3)
        except (TypeError, ValueError, KeyError) as e:
            safe_print(f"Tournament selection failed: {e}")
            tournament_selected = []

        # New population
        population = elites + tournament_selected

        # If population size reduced due to failures, pad with top remaining candidates
        if len(population) < POPULATION_SIZE:
            # sort combined_population by combined_score descending
            try:
                sorted_candidates = sorted(combined_population, key=lambda ind: float(ind.get("fitness", {}).get("combined_score", 0.0)), reverse=True)
            except (TypeError, ValueError) as e:
                sorted_candidates = combined_population
            idx_pad = 0
            while len(population) < POPULATION_SIZE and idx_pad < len(sorted_candidates):
                candidate = sorted_candidates[idx_pad]
                if candidate not in population:
                    population.append(candidate)
                idx_pad += 1

        # Update archive with best individuals from combined set
        try:
            archive = archive_best_individuals(archive, combined_population, max_archive_size=20)
        except (TypeError, ValueError) as e:
            safe_print(f"Archive update failed: {e}")

        # Track best score and stagnation
        try:
            current_best = max([float(ind.get("fitness", {}).get("combined_score", 0.0)) for ind in population])
        except (TypeError, ValueError) as e:
            current_best = best_combined_score

        best_history.append(current_best)
        if current_best > best_combined_score + 1e-12:
            best_combined_score = current_best
            stagnation_count = 0
            safe_print(f"New best combined_score in generation {gen}: {best_combined_score}")
        else:
            stagnation_count += 1
            safe_print(f"No improvement this generation (stagnation_count={stagnation_count})")

        # Print brief population summary
        safe_print("Population summary (top entries):")
        for i, ind in enumerate(sorted(population, key=lambda x: float(x.get("fitness", {}).get("combined_score", 0.0)), reverse=True)[:5], start=1):
            safe_print(f"  Rank {i}: {_summarize_individual(ind)}")

        if output_run_dir is not None:
            save_generation_snapshot(
                output_run_dir,
                generation=gen,
                label=f"after_gen{gen}",
                population=population,
                archive=archive,
                extra={
                    "strategies_used": strategies_used,
                    "offspring_raw_count": len(offspring_raw_contents),
                    "offspring_evaluated_count": len(valid_offspring),
                    "offspring_failed_count": invalid_offspring_count,
                    "best_combined_score": best_combined_score,
                    "current_best_in_population": current_best,
                    "stagnation_count": stagnation_count,
                },
            )

    # End of generations: report best archive entry
    if archive:
        top_archived = archive[0]
        safe_print("\n=== Evolution complete ===")
        safe_print("Top archived individual summary:")
        safe_print(f"Thought: {top_archived.get('thought')}")
        safe_print(f"Fitness: {top_archived.get('fitness')}")
        safe_print("Code (truncated):")
        code_text = top_archived.get("code", "")
        safe_print(code_text + ("..." if len(code_text) > 1000 else ""))
    else:
        safe_print("No archived individuals available at end of run.")

    if output_run_dir is not None:
        save_final_archive(output_run_dir, archive)
        if archive:
            save_code_artifact(
                output_run_dir,
                filename="best_code.py",
                code=archive[0].get("code", ""),
            )
    return archive


def run_full_test_for_archive(
    task: EvolutionTask,
    archive: List[Dict[str, Any]],
    *,
    mode: str,
    output_run_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not archive:
        safe_print("Skipping full test: archive is empty.")
        return None
    top_archived = archive[0]
    safe_print(f"\n=== Running full {mode} test for archived best individual ===")
    code = top_archived.get("code", "")
    code_file: Optional[Path] = None
    if output_run_dir is not None:
        code_file = save_code_artifact(
            output_run_dir,
            filename="best_code.py",
            code=code,
        )
    try:
        full_test = task.run_full_test(code, mode=mode)
    except NotImplementedError as e:
        safe_print(str(e))
        return None
    payload: Dict[str, Any] = {
        "task_id": task.id,
        "target_function_name": task.target_function_name,
        "mode": mode,
        "archive_fitness": top_archived.get("fitness", {}),
        "full_test": full_test,
    }
    if code_file is not None:
        payload["code_file"] = str(code_file.resolve())
    if full_test.get("error"):
        safe_print(f"Full test failed: {full_test.get('error')}")
    else:
        for problem_size, metrics in full_test.get("problem_sizes", {}).items():
            safe_print(f"  Problem size {problem_size}: {_format_full_test_metrics(metrics)}")
        if "mean_gap_percent" in full_test:
            safe_print(f"Full test mean gap: {full_test.get('mean_gap_percent')}%")
        elif "mean_combined_score" in full_test:
            safe_print(f"Full test mean combined_score: {full_test.get('mean_combined_score')}")
    if output_run_dir is not None:
        save_final_test_result(output_run_dir, payload)
        safe_print(f"Saved full test result to: {output_run_dir / 'final_test.json'}")
    return payload


def run_additional_test_for_archive(
    task: EvolutionTask,
    archive: List[Dict[str, Any]],
    *,
    label: str,
    output_run_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not archive:
        return None
    top_archived = archive[0]
    safe_print(f"\n=== Running additional `{label}` test for archived best individual ===")
    code = top_archived.get("code", "")
    try:
        extra_test = task.run_additional_test(code, label=label)
    except NotImplementedError:
        return None
    payload: Dict[str, Any] = {
        "task_id": task.id,
        "target_function_name": task.target_function_name,
        "label": label,
        "archive_fitness": top_archived.get("fitness", {}),
        "additional_test": extra_test,
    }
    if extra_test.get("error"):
        safe_print(f"Additional `{label}` test failed: {extra_test.get('error')}")
    elif label == "tsplib":
        safe_print(f"Additional `{label}` mean gap: {extra_test.get('mean_gap_percent')}%")
    if output_run_dir is not None:
        filename = f"additional_test_{label}.json"
        save_named_result(output_run_dir, filename=filename, result=payload)
        safe_print(f"Saved additional `{label}` result to: {output_run_dir / filename}")
    return payload


def run_full_test_for_code_path(
    task: EvolutionTask,
    code_path: Path,
    *,
    mode: str,
    output_run_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not code_path.is_file():
        raise FileNotFoundError(f"Code file not found: {code_path}")
    code = code_path.read_text(encoding="utf-8")
    safe_print(f"\n=== Running full {mode} test for code file: {code_path} ===")
    copied_code_path: Optional[Path] = None
    if output_run_dir is not None:
        copied_code_path = save_code_artifact(
            output_run_dir,
            filename="tested_code.py",
            code=code,
        )
    try:
        full_test = task.run_full_test(code, mode=mode)
    except NotImplementedError as e:
        safe_print(str(e))
        return None
    payload: Dict[str, Any] = {
        "task_id": task.id,
        "target_function_name": task.target_function_name,
        "mode": mode,
        "source_code_path": str(code_path.resolve()),
        "full_test": full_test,
    }
    if copied_code_path is not None:
        payload["copied_code_file"] = str(copied_code_path.resolve())
    if full_test.get("error"):
        safe_print(f"Standalone full test failed: {full_test.get('error')}")
    else:
        for problem_size, metrics in full_test.get("problem_sizes", {}).items():
            safe_print(f"  Problem size {problem_size}: {_format_full_test_metrics(metrics)}")
        if "mean_gap_percent" in full_test:
            safe_print(f"Full test mean gap: {full_test.get('mean_gap_percent')}%")
        elif "mean_combined_score" in full_test:
            safe_print(f"Full test mean combined_score: {full_test.get('mean_combined_score')}")
    if output_run_dir is not None:
        save_final_test_result(output_run_dir, payload)
        safe_print(f"Saved full test result to: {output_run_dir / 'final_test.json'}")
    return payload


def run_additional_test_for_code_path(
    task: EvolutionTask,
    code_path: Path,
    *,
    label: str,
    output_run_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not code_path.is_file():
        raise FileNotFoundError(f"Code file not found: {code_path}")
    code = code_path.read_text(encoding="utf-8")
    safe_print(f"\n=== Running additional `{label}` test for code file: {code_path} ===")
    try:
        extra_test = task.run_additional_test(code, label=label)
    except NotImplementedError:
        return None
    payload: Dict[str, Any] = {
        "task_id": task.id,
        "target_function_name": task.target_function_name,
        "label": label,
        "source_code_path": str(code_path.resolve()),
        "additional_test": extra_test,
    }
    if extra_test.get("error"):
        safe_print(f"Additional `{label}` test failed: {extra_test.get('error')}")
    elif label == "tsplib":
        safe_print(f"Additional `{label}` mean gap: {extra_test.get('mean_gap_percent')}%")
    if output_run_dir is not None:
        filename = f"additional_test_{label}.json"
        save_named_result(output_run_dir, filename=filename, result=payload)
        safe_print(f"Saved additional `{label}` result to: {output_run_dir / filename}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM + 演化框架（多任务通过 --task 切换）")
    parser.add_argument(
        "--task",
        required=True,
        help=f"任务 ID。已注册: {', '.join(list_tasks())}",
    )
    parser.add_argument(
        "--llm-preset",
        choices=["ollama", "openai", "deepseek", "agicto", "vllm_qwen3"],
        default=None,
        help="快捷预设：ollama / openai / deepseek / agicto；vllm_qwen3=本机10008（model 与 /v1/models 的 id 一致）",
    )
    parser.add_argument("--llm-config", default=None, metavar="PATH", help="JSON 配置文件路径（写入 LLM_CONFIG_FILE）")
    parser.add_argument("--llm-base-url", default=None, help="覆盖 LLM_BASE_URL，如 http://127.0.0.1:11434/v1")
    parser.add_argument("--llm-api-key", default=None, help="覆盖 LLM_API_KEY / OPENAI_API_KEY")
    parser.add_argument("--llm-model", default=None, help="覆盖 LLM_MODEL")
    parser.add_argument("--llm-timeout", type=float, default=None, help="覆盖 LLM_TIMEOUT（秒）")
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=DEFAULT_LLM_CONCURRENCY,
        help="LLM 并发请求数（初始化种群与每代 offspring 共用）",
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=DEFAULT_EVAL_CONCURRENCY,
        help="每代 offspring 代码评测的多进程并发数",
    )
    parser.add_argument("--generations", type=int, default=GENERATIONS, help="覆盖进化代数")
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE, help="覆盖种群大小")
    parser.add_argument("--offspring-per-gen", type=int, default=OFFSPRING_PER_GEN, help="覆盖每代后代数")
    parser.add_argument("--elitism-count", type=int, default=ELITISM_COUNT, help="覆盖精英保留数")
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=DEFAULT_EVAL_TIMEOUT_SECONDS,
        metavar="SEC",
        help="单个个体的代码评测墙钟超时（秒）；0 或负数关闭超时（恢复原先无限制行为）",
    )
    parser.add_argument(
        "--no-run-output",
        action="store_true",
        help="不创建 output/<task>/<时间戳>/ 目录（默认每次运行都会记录终端与演化快照）",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="指定单个物理 GPU 编号，例如 `--gpu 1`；会映射为进程内的 `cuda:0`",
    )
    gpu_group.add_argument(
        "--gpus",
        default=None,
        metavar="IDS",
        help="指定多个物理 GPU 编号，例如 `--gpus 5,6`；评测 worker 会按逻辑设备轮转",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="演化结束后，对最终 archive 第一名运行完整测试，并写入 final_test.json",
    )
    parser.add_argument(
        "--full-test-mode",
        choices=["val", "test"],
        default="test",
        help="`--full-test` 使用的完整评测配置，默认 `test`",
    )
    parser.add_argument(
        "--test-code",
        default=None,
        metavar="PATH",
        help="跳过演化，直接对指定 `.py` 代码文件运行完整测试",
    )
    args = parser.parse_args()

    POPULATION_SIZE = max(2, int(args.population_size))
    GENERATIONS = max(1, int(args.generations))
    OFFSPRING_PER_GEN = max(1, int(args.offspring_per_gen))
    ELITISM_COUNT = max(1, min(int(args.elitism_count), POPULATION_SIZE))

    import os

    gpu_selection: int | list[int] | None
    if args.gpus:
        gpu_selection = [int(part.strip()) for part in args.gpus.split(",") if part.strip()]
    else:
        gpu_selection = args.gpu
    configure_gpu_environment(gpu_selection)
    eval_gpu_logical_ids = list(range(len(visible_gpu_ids()))) if gpu_requested() else None
    eval_timeout_param: float | None = None if args.eval_timeout <= 0 else float(args.eval_timeout)
    standalone_test = args.test_code is not None
    if args.llm_preset:
        os.environ["LLM_PRESET"] = args.llm_preset
    if args.llm_config:
        os.environ["LLM_CONFIG_FILE"] = args.llm_config
    if not standalone_test:
        configure_llm(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            timeout=args.llm_timeout,
            replace_all=True,
        )

    eoa_root = Path(__file__).resolve().parent
    run_dir: Optional[Path] = None
    task = get_task(args.task)
    persist_run_output = (not args.no_run_output) or args.full_test or standalone_test
    if args.no_run_output and (args.full_test or standalone_test):
        print("[info] full test output requires persisted output; creating output directory anyway.", flush=True)
    if persist_run_output:
        run_dir = make_run_output_dir(eoa_root, args.task)
        if standalone_test:
            llm_meta = {"skipped": True, "reason": "standalone_full_test"}
        else:
            s = get_llm_settings()
            key = s.api_key
            if not key:
                masked = "(empty)"
            elif len(key) <= 8:
                masked = "***"
            else:
                masked = key[:4] + "…" + key[-2:]
            llm_meta = {
                "base_url": s.base_url,
                "model": s.model,
                "timeout": s.timeout,
                "api_key_preview": masked,
            }
        write_run_meta(
            run_dir,
            task_id=args.task,
            target_function_name=task.target_function_name,
            argv=sys.argv,
            llm_meta=llm_meta,
            runtime_meta={
                "gpu_requested": gpu_requested(),
                "visible_gpu": visible_gpu() or None,
                "eval_timeout_seconds": eval_timeout_param,
            },
        )

    start = time.time()
    try:
        if run_dir is not None:
            with tee_terminal_to_file(run_dir):
                if standalone_test:
                    run_full_test_for_code_path(
                        task,
                        Path(args.test_code),
                        mode=args.full_test_mode,
                        output_run_dir=run_dir,
                    )
                    run_additional_test_for_code_path(
                        task,
                        Path(args.test_code),
                        label="tsplib",
                        output_run_dir=run_dir,
                    )
                else:
                    archive = main(
                        task,
                        output_run_dir=run_dir,
                        llm_concurrency=args.llm_concurrency,
                        eval_concurrency=args.eval_concurrency,
                        eval_gpu_logical_ids=eval_gpu_logical_ids,
                        eval_timeout_seconds=eval_timeout_param,
                    )
                    if args.full_test:
                        run_full_test_for_archive(
                            task,
                            archive,
                            mode=args.full_test_mode,
                            output_run_dir=run_dir,
                        )
                    run_additional_test_for_archive(
                        task,
                        archive,
                        label="tsplib",
                        output_run_dir=run_dir,
                    )
                end = time.time()
                safe_print(f"\nTotal runtime: {end - start:.2f} seconds")
        else:
            if standalone_test:
                run_full_test_for_code_path(
                    task,
                    Path(args.test_code),
                    mode=args.full_test_mode,
                    output_run_dir=None,
                )
                run_additional_test_for_code_path(
                    task,
                    Path(args.test_code),
                    label="tsplib",
                    output_run_dir=None,
                )
            else:
                archive = main(
                    task,
                    output_run_dir=None,
                    llm_concurrency=args.llm_concurrency,
                    eval_concurrency=args.eval_concurrency,
                    eval_gpu_logical_ids=eval_gpu_logical_ids,
                    eval_timeout_seconds=eval_timeout_param,
                )
                if args.full_test:
                    run_full_test_for_archive(
                        task,
                        archive,
                        mode=args.full_test_mode,
                        output_run_dir=None,
                    )
                run_additional_test_for_archive(
                    task,
                    archive,
                    label="tsplib",
                    output_run_dir=None,
                )
            end = time.time()
            safe_print(f"\nTotal runtime: {end - start:.2f} seconds")
    except KeyboardInterrupt:
        cleanup_process_pool()
        safe_print("\nInterrupted by user. Cleaned up worker processes.")
        raise SystemExit(130)

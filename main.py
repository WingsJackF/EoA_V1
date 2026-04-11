import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
sys.path.append(project_root)
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
from tasks.base import EvolutionTask
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
from run_output_recorder import (
    make_run_output_dir,
    save_final_archive,
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


def _summarize_individual(ind: Dict[str, Any]) -> str:
    """Return a concise string summarizing an individual for logging."""
    cs = None
    try:
        cs = ind.get("fitness", {}).get("combined_score")
    except Exception:
        cs = None
    thought = (ind.get("thought") or "")[:120]
    return f"combined_score={cs}, thought={thought}"


def main(
    task: EvolutionTask,
    *,
    output_run_dir: Optional[Path] = None,
    llm_concurrency: int = DEFAULT_LLM_CONCURRENCY,
) -> None:
    """
    Main evolutionary loop orchestration.
    """
    # Initialize population with seed + LLM-generated diverse individuals
    if output_run_dir is not None:
        safe_print(f"Run output directory: {output_run_dir.resolve()}")
    safe_print(f"Task: {task.id} (entry: {task.target_function_name})")
    _log_llm_settings()
    if llm_concurrency <= 0:
        raise ValueError("llm_concurrency must be a positive integer")
    safe_print(f"Initializing population with seed individual (llm_concurrency={llm_concurrency})...")
    try:
        seed_ind = add_seed_individual(task)
    except (SyntaxError, ValueError, TypeError) as e:
        safe_print(f"Seed validation/evaluation failed: {e}")
        return

    population: List[Dict[str, Any]] = [seed_ind]
    # Generate remaining individuals using the LLM interaction module
    n_to_generate = POPULATION_SIZE - 1
    if n_to_generate > 0:
        try:
            generated = generate_diverse_initial_individuals(
                task,
                n_to_generate,
                max_concurrency=llm_concurrency,
            )
            # Only append properly formatted dicts
            for ind in generated:
                if isinstance(ind, dict):
                    population.append(ind)
        except ImportError as imp_err:
            safe_print(f"LLM generation module import failed: {imp_err}")
            # Fall back: let the task provide local fallback individuals.
            for s in range(n_to_generate):
                try:
                    population.append(task.build_fallback_individual(s))
                except (TypeError, ImportError, ValueError, SyntaxError) as eval_err:
                    population.append(
                        {
                            "thought": f"Fallback generation failed #{s + 1}",
                            "code": "",
                            "fitness": {
                                "min_max_ratio": 0.0,
                                "combined_score": 0.0,
                                "eval_time": 0.0,
                                "error": str(eval_err),
                            },
                        }
                    )

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

        request_results = implement_post_requests_concurrently(
            [payload for _, payload in prepared_offspring_requests],
            max_retries=5,
            base_backoff=2.0,
            timeout=30.0,
            verbose=False,
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

        # Integrate offspring: parse, validate, evaluate
        try:
            evaluated_offspring = collect_and_integrate_offspring_results(offspring_raw_contents, task)
        except ImportError as ie:
            safe_print(f"Integration failed due to missing modules: {ie}")
            evaluated_offspring = []
        except TypeError as te:
            safe_print(f"Integration type error: {te}")
            evaluated_offspring = []

        safe_print(f"Evaluated offspring count: {len(evaluated_offspring)}")
        for idx, off in enumerate(evaluated_offspring, start=1):
            safe_print(f"  Offspring {idx}: {_summarize_individual(off)}")

        # Combine population and offspring for selection
        combined_population = population + evaluated_offspring

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
                    "offspring_evaluated_count": len(evaluated_offspring),
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
        "--no-run-output",
        action="store_true",
        help="不创建 output/<task>/<时间戳>/ 目录（默认每次运行都会记录终端与演化快照）",
    )
    args = parser.parse_args()

    import os

    if args.llm_preset:
        os.environ["LLM_PRESET"] = args.llm_preset
    if args.llm_config:
        os.environ["LLM_CONFIG_FILE"] = args.llm_config
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
    if not args.no_run_output:
        run_dir = make_run_output_dir(eoa_root, args.task)
        s = get_llm_settings()
        key = s.api_key
        if not key:
            masked = "(empty)"
        elif len(key) <= 8:
            masked = "***"
        else:
            masked = key[:4] + "…" + key[-2:]
        write_run_meta(
            run_dir,
            task_id=args.task,
            target_function_name=task.target_function_name,
            argv=sys.argv,
            llm_meta={
                "base_url": s.base_url,
                "model": s.model,
                "timeout": s.timeout,
                "api_key_preview": masked,
            },
        )

    start = time.time()
    if run_dir is not None:
        with tee_terminal_to_file(run_dir):
            main(task, output_run_dir=run_dir, llm_concurrency=args.llm_concurrency)
            end = time.time()
            safe_print(f"\nTotal runtime: {end - start:.2f} seconds")
    else:
        main(task, output_run_dir=None, llm_concurrency=args.llm_concurrency)
        end = time.time()
        safe_print(f"\nTotal runtime: {end - start:.2f} seconds")

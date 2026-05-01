"""Console reporting helpers for task evaluations."""

from __future__ import annotations

from typing import Any, Mapping


def print_full_test_problem_result(problem_size: Any, metrics: Mapping[str, Any]) -> None:
    """Print one completed full-test result in a ReEvo-like style."""
    if {"teacher_score", "student_score", "gap_percent"}.issubset(metrics):
        print(
            f"Problem size: {problem_size}, "
            f"Optimal: {metrics['teacher_score']}, "
            f"Student: {metrics['student_score']}, "
            f"Gap: {metrics['gap_percent']}",
            flush=True,
        )
        return

    if "objective" in metrics:
        print(
            f"Problem size: {problem_size}, "
            f"Objective: {metrics['objective']}, "
            f"Combined score: {metrics.get('combined_score')}",
            flush=True,
        )
        return

    if {"avg_bins", "l1_bound", "excess_percent"}.issubset(metrics):
        print(
            f"Problem size: {problem_size}, "
            f"Avg bins: {metrics['avg_bins']}, "
            f"L1 bound: {metrics['l1_bound']}, "
            f"Excess: {metrics['excess_percent']}%, "
            f"Combined score: {metrics.get('combined_score')}",
            flush=True,
        )
        return

    if "average_reward" in metrics:
        print(
            f"Problem size: {problem_size}, "
            f"Average reward: {metrics['average_reward']}, "
            f"Combined score: {metrics.get('combined_score')}",
            flush=True,
        )
        return

    if "min_max_ratio" in metrics:
        print(
            f"Problem size: {problem_size}, "
            f"Min/max ratio: {metrics['min_max_ratio']}, "
            f"Combined score: {metrics.get('combined_score')}",
            flush=True,
        )
        return

    print(f"Problem size: {problem_size}, Metrics: {dict(metrics)}", flush=True)

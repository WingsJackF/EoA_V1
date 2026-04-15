"""任务专用评估：MKP ACO。"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import numpy as np
import torch

from tasks.task_support.eval_timeout import run_spawn_eval_jobs
from tasks.task_support.gpu import gpu_requested, solver_device, visible_gpu_ids
from tasks.task_support.paths import problem_dir
from tasks.task_support.processes import cleanup_process_pool
from tasks.task_support.runtime import import_problem_module, load_program_module, resolve_callable

MAX_INSTANCES = 5
N_ITERATIONS = 50
N_ANTS = 10
POSSIBLE_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")
FULL_TEST_CHUNK_SIZE = 8
FULL_TEST_CHUNK_TIMEOUT_SECONDS = 3600.0
FULL_TEST_SIZES = {
    "val": (100, 300, 500),
    "test": (100, 200, 300, 500, 1000),
}


def _wrap(ok: Callable[[], Any]) -> Dict[str, Any]:
    try:
        raw = ok()
        return {"combined_score": float(raw), "eval_time": 0.0, "error": None}
    except Exception as e:
        return {"combined_score": 0.0, "eval_time": 0.0, "error": str(e)}


def _mkp_objective_for_instance(
    heuristics: Callable[..., Any],
    aco_mod: Any,
    prize: np.ndarray,
    weight: np.ndarray,
) -> float:
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    if tuple(heu.shape) != (weight.shape[0],):
        raise ValueError(f"Invalid heuristic shape: {heu.shape}")
    heu[heu < 1e-9] = 1e-9
    solver = aco_mod.ACO(
        torch.from_numpy(prize),
        torch.from_numpy(weight),
        torch.from_numpy(heu),
        N_ANTS,
        device=solver_device(),
    )
    obj, _ = solver.run(N_ITERATIONS)
    return float(obj.item() if hasattr(obj, "item") else obj)


def _chunk_ranges(total: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(total, start + chunk_size)


def _evaluate_full_test_chunk(
    program_code: str,
    dataset_path: str,
    start: int,
    stop: int,
    logical_cuda_device: int | None = None,
) -> Dict[str, Any]:
    from tasks.task_support.gpu import configure_logical_cuda_device

    configure_logical_cuda_device(logical_cuda_device)
    module = load_program_module(program_code, module_name="mkp_aco_full_candidate_worker")
    heuristics = resolve_callable(module, POSSIBLE_NAMES)
    base = problem_dir("mkp_aco")
    aco_mod = import_problem_module(base, "aco")
    data = np.load(dataset_path)
    prizes, weights = data["prizes"], data["weights"]
    objs = [
        _mkp_objective_for_instance(heuristics, aco_mod, prizes[i], weights[i])
        for i in range(start, stop)
    ]
    return {"start": start, "stop": stop, "objs": objs, "error": None}


def spawn_mkp_full_test_worker(
    program_code: str,
    dataset_path: str,
    start: int,
    stop: int,
    logical_cuda_device: int | None,
    out_q: Any,
) -> None:
    try:
        out_q.put(_evaluate_full_test_chunk(program_code, dataset_path, start, stop, logical_cuda_device))
    except Exception as exc:  # noqa: BLE001
        out_q.put({"start": start, "stop": stop, "objs": [], "error": f"Full-test worker crashed: {exc!r}"})


def run_evaluation(program_code: str) -> Dict[str, Any]:
    def inner() -> float:
        module = load_program_module(program_code, module_name="mkp_aco_candidate")
        heuristics = resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("mkp_aco")
        aco_mod = import_problem_module(base, "aco")
        ds = base / "dataset" / "train100_dataset.npz"
        if not ds.is_file():
            ds = base / "dataset" / "train50_dataset.npz"
        if not ds.is_file():
            raise FileNotFoundError(f"缺少数据集 {base / 'dataset'} 下的 train100/train50 数据")
        data = np.load(ds)
        prizes, weights = data["prizes"], data["weights"]
        k = min(MAX_INSTANCES, prizes.shape[0])
        objs = []
        for i in range(k):
            objs.append(_mkp_objective_for_instance(heuristics, aco_mod, prizes[i], weights[i]))
        return float(np.mean(objs))

    return _wrap(inner)


def run_full_test(program_code: str, *, mode: str = "test") -> Dict[str, Any]:
    try:
        if mode not in FULL_TEST_SIZES:
            raise ValueError(f"Unsupported full test mode: {mode}")
        module = load_program_module(program_code, module_name="mkp_aco_full_candidate")
        resolve_callable(module, POSSIBLE_NAMES)
        base = problem_dir("mkp_aco")
        train_ds = base / "dataset" / "train100_dataset.npz"
        if not train_ds.is_file():
            gen_inst_mod = import_problem_module(base, "gen_inst")
            gen_inst_mod.generate_datasets()
        size_paths: Dict[int, Path] = {}
        size_counts: Dict[str, int] = {}
        for problem_size in FULL_TEST_SIZES[mode]:
            ds = base / "dataset" / f"{mode}{problem_size}_dataset.npz"
            if not ds.is_file():
                raise FileNotFoundError(f"缺少数据集 {ds}")
            data = np.load(ds)
            size_paths[problem_size] = ds
            size_counts[str(problem_size)] = int(data["prizes"].shape[0])
        size_to_objective: Dict[str, Dict[str, float]] = {}
        logical_cuda_devices = list(range(len(visible_gpu_ids()))) if gpu_requested() and visible_gpu_ids() else []
        jobs: list[tuple[int, tuple[Any, ...]]] = []
        job_meta: Dict[int, Dict[str, Any]] = {}
        job_idx = 0
        overall_total = sum(size_counts.values())
        for problem_size in FULL_TEST_SIZES[mode]:
            total = size_counts[str(problem_size)]
            print(
                f"[mkp_aco full test] queued size={problem_size} instances={total} chunk_size={FULL_TEST_CHUNK_SIZE}",
                flush=True,
            )
            for start, stop in _chunk_ranges(total, FULL_TEST_CHUNK_SIZE):
                logical_cuda_device = None
                if logical_cuda_devices:
                    logical_cuda_device = logical_cuda_devices[job_idx % len(logical_cuda_devices)]
                jobs.append((job_idx, (program_code, str(size_paths[problem_size]), start, stop, logical_cuda_device)))
                job_meta[job_idx] = {
                    "problem_size": str(problem_size),
                    "start": start,
                    "stop": stop,
                }
                job_idx += 1
        if logical_cuda_devices:
            max_workers = min(len(jobs), len(logical_cuda_devices))
            worker_desc = ", ".join(f"cuda:{device}" for device in logical_cuda_devices)
        else:
            max_workers = min(len(jobs), max(1, min((mp.cpu_count() or 1), 4)))
            worker_desc = f"{max_workers} CPU workers"
        print(
            f"[mkp_aco full test] starting {len(jobs)} chunks across {max_workers} workers ({worker_desc})",
            flush=True,
        )
        size_results: Dict[str, list[float]] = {str(problem_size): [] for problem_size in FULL_TEST_SIZES[mode]}
        completed_counts: Dict[str, int] = {str(problem_size): 0 for problem_size in FULL_TEST_SIZES[mode]}
        overall_done = 0

        def _on_job_complete(idx: int, payload: Dict[str, Any]) -> None:
            nonlocal overall_done
            meta = job_meta[idx]
            problem_size = meta["problem_size"]
            if payload.get("error"):
                print(
                    f"[mkp_aco full test] size={problem_size} chunk={meta['start']}:{meta['stop']} failed: {payload['error']}",
                    flush=True,
                )
                return
            chunk_count = int(payload["stop"]) - int(payload["start"])
            completed_counts[problem_size] += chunk_count
            overall_done += chunk_count
            print(
                f"[mkp_aco full test] size={problem_size} progress {completed_counts[problem_size]}/{size_counts[problem_size]} "
                f"| overall {overall_done}/{overall_total}",
                flush=True,
            )

        try:
            results_map = run_spawn_eval_jobs(
                jobs,
                max_workers=max_workers,
                job_timeout_seconds=FULL_TEST_CHUNK_TIMEOUT_SECONDS,
                worker_target=spawn_mkp_full_test_worker,
                on_job_complete=_on_job_complete,
            )
        except KeyboardInterrupt:
            cleanup_process_pool(None)
            raise

        for idx in sorted(results_map):
            payload = results_map[idx]
            meta = job_meta[idx]
            if payload.get("error"):
                raise RuntimeError(
                    f"problem_size={meta['problem_size']} chunk={meta['start']}:{meta['stop']} failed: {payload['error']}"
                )
            size_results[meta["problem_size"]].extend(payload["objs"])

        for problem_size in FULL_TEST_SIZES[mode]:
            objs = size_results[str(problem_size)]
            avg_obj = float(np.mean(objs))
            size_to_objective[str(problem_size)] = {
                "objective": avg_obj,
                "combined_score": avg_obj,
            }
        mean_combined_score = float(np.mean([x["combined_score"] for x in size_to_objective.values()]))
        return {"mode": mode, "problem_sizes": size_to_objective, "mean_combined_score": mean_combined_score, "error": None}
    except Exception as e:
        return {"mode": mode, "problem_sizes": {}, "mean_combined_score": None, "error": str(e)}

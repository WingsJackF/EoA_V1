#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    "no-exploration",
    "no-simplification",
    "no-adaptive",
]

TASKS = [
    "cvrp_lehd",
    "cvrp_pomo",
    "tsp_lehd",
    "tsp_pomo",
]

COMMON_ARGS = [
    "--llm-concurrency",
    "8",
    "--eval-concurrency",
    "2",
    "--full-test",
    "--gpu",
    "4",
    "--full-test-mode",
    "val",
]


def build_commands(python_executable: str, output_root: Path | None = None) -> list[list[str]]:
    script_dir = Path(__file__).resolve().parent
    main_py = script_dir / "main.py"
    commands: list[list[str]] = []
    for ablation in ABLATIONS:
        for task in TASKS:
            command = [
                python_executable,
                str(main_py),
                "--task",
                task,
                "--strategy-ablation",
                ablation,
                *COMMON_ARGS,
            ]
            if output_root is not None:
                command.extend(["--output-dir", str(output_root / ablation / task)])
            commands.append(command)
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strategy ablation experiments for CVRP/TSP LEHD/POMO tasks."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run EoA_V1/main.py. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later experiments if one command fails.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        metavar="PATH",
        help="Optional root directory for experiment outputs; each run uses <root>/<ablation>/<task>.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser() if args.output_root else None
    commands = build_commands(args.python, output_root=output_root)
    total = len(commands)
    failures: list[tuple[int, list[str], int]] = []

    for index, command in enumerate(commands, start=1):
        print("=" * 88, flush=True)
        print(f"[{index}/{total}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent)
        if result.returncode != 0:
            failures.append((index, command, result.returncode))
            print(
                f"[failed] Experiment {index}/{total} exited with code {result.returncode}",
                flush=True,
            )
            if not args.continue_on_error:
                return result.returncode

    if failures:
        print("=" * 88, flush=True)
        print(f"Completed with {len(failures)} failed experiment(s):", flush=True)
        for index, command, returncode in failures:
            print(f"  [{index}] returncode={returncode}: {' '.join(command)}", flush=True)
        return 1

    print("=" * 88, flush=True)
    print(f"Completed {total} experiment(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""从项目内置提示词资产目录读取文本，生成兼容当前演化算子的 prompt 三件套。"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

from tasks.task_support.paths import prompt_dir


def load_prompt_assets(prompt_key: str) -> Dict[str, str]:
    base = prompt_dir(prompt_key)

    def _read(name: str) -> str:
        p = base / name
        return p.read_text(encoding="utf-8").strip() if p.is_file() else ""

    return {
        "seed_func": _read("seed_func.txt"),
        "func_signature": _read("func_signature.txt"),
        "func_desc": _read("func_desc.txt"),
        "external_knowledge": _read("external_knowledge.txt"),
    }


def build_prompt_strategies(assets: Dict[str, str]) -> Dict[str, Callable[..., Tuple[str, str]]]:
    problem_block = "\n".join(
        [
            "## Problem (func_desc)",
            assets.get("func_desc", "(no func_desc)"),
            "",
            "## Required signature template",
            assets.get("func_signature", ""),
            "",
            "## Reference seed implementation",
            "```python",
            assets.get("seed_func", "")[:8000],
            "```",
        ]
    )
    ext = assets.get("external_knowledge", "").strip()
    if ext:
        problem_block += "\n\n## External knowledge\n" + ext[:6000]

    def _fmt_response() -> str:
        return (
            "\n\nResponse format (required):\n"
            "Strategic Thought: <one or two sentences>\n\n"
            "```python\n# full runnable code\n```\n"
        )

    def modification(parent_individual: Dict[str, str]) -> Tuple[str, str]:
        system = (
            "You are an expert in combinatorial optimization and Python. "
            "Improve the given heuristic/operator code with targeted, verifiable changes. "
            "Preserve the core idea unless clearly inferior."
        )
        thought = parent_individual.get("thought") or "(none)"
        code = parent_individual.get("code") or ""
        user = (
            problem_block
            + "\n\n## Parent\nStrategic Thought:\n"
            + str(thought)
            + "\n\nCode:\n```python\n"
            + str(code)
            + "\n```\n\n## Task\nApply a **modification** strategy: small but meaningful improvement.\n"
            + _fmt_response()
        )
        return system, user

    def exploration(parents: List[Dict[str, str]]) -> Tuple[str, str]:
        system = (
            "You are an expert in combinatorial optimization. **Synthesize** novel code from multiple parents, "
            "merging strengths; avoid trivial stitching."
        )
        parts = [problem_block, "\n## Parents for synthesis\n"]
        for i, p in enumerate(parents, start=1):
            parts.append(f"### Parent {i}\nThought: {p.get('thought','')}\n```python\n{p.get('code','')}\n```\n")
        parts.append("\n## Task\nProduce one new implementation combining ideas.\n" + _fmt_response())
        return system, "\n".join(parts)

    def simplification(parent_individual: Dict[str, str]) -> Tuple[str, str]:
        system = (
            "You simplify optimization code: shorter, clearer, numerically stable, same or better behavior."
        )
        thought = parent_individual.get("thought") or "(none)"
        code = parent_individual.get("code") or ""
        user = (
            problem_block
            + "\n\n## Parent to simplify\nThought:\n"
            + str(thought)
            + "\n\nCode:\n```python\n"
            + str(code)
            + "\n```\n\n## Task\nSimplify while keeping the interface implied by the signature template.\n"
            + _fmt_response()
        )
        return system, user

    return {
        "modification": modification,
        "exploration": exploration,
        "simplification": simplification,
    }


def initial_system_prompt(assets: Dict[str, str], task_label: str) -> str:
    return (
        f"You are an expert in combinatorial optimization ({task_label}). "
        "Generate Python code for the problem described below. "
        "Return Strategic Thought plus a single fenced python block with the full solution.\n\n"
        + assets.get("func_desc", "")
    )


def diversity_lines_from_assets(assets: Dict[str, str]) -> List[str]:
    sig = assets.get("func_signature", "").replace("{version}", "2")
    base = [
        "Invent a clearly different heuristic strategy from the seed.",
        "Emphasize exploitation of local distance structure.",
        "Emphasize diversification / exploration across nodes.",
        "Try a hybrid that balances greedy scoring with lookahead or penalty terms.",
        "Use a simpler formula with fewer hyperparameters than the seed.",
    ]
    if sig:
        base.append(f"Strictly respect a callable shaped like: {sig[:200]}...")
    return base

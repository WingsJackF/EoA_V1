"""演化任务：CVRP POMO-style torch heuristics"""

from __future__ import annotations

import ast
from typing import Any, Dict, List

from tasks.base import EvolutionTask, PromptStrategies
from tasks.task_support.text_utils import first_function_name

from . import evaluation, prompts, seed
from .constants import PROBLEM_SUBDIR, TASK_ID


class CvrpPomoTask(EvolutionTask):
    id = TASK_ID

    def __init__(self) -> None:
        self._seed_code = seed.get_seed_code()
        self.target_function_name = first_function_name(self._seed_code)

    @property
    def seed_code(self) -> str:
        return self._seed_code.strip()

    @property
    def seed_thought(self) -> str:
        fd = prompts.PROMPT_ASSETS.get("func_desc", "").strip()
        return (fd[:400] + "…") if len(fd) > 400 else (fd or f"Task: {TASK_ID}")

    def validate_syntax(self, code: str) -> None:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        ast.parse(code)

    def evaluate_raw(self, code: str) -> Dict[str, Any]:
        return evaluation.run_evaluation(code)

    @property
    def prompt_strategies(self) -> PromptStrategies:
        return prompts.PROMPT_STRATEGIES

    @property
    def initial_population_system_prompt(self) -> str:
        return prompts.INITIAL_POPULATION_SYSTEM_PROMPT

    @property
    def diversity_instructions(self) -> List[str]:
        return list(prompts.DIVERSITY_INSTRUCTIONS)

    def format_diversity_user_prompt(self, diversity_instruction: str) -> str:
        sig = prompts.PROMPT_ASSETS.get("func_signature", "").replace("{version}", "2")
        return "".join(
            [
                "Reference task description and signature:\n\n",
                prompts.PROMPT_ASSETS.get("func_desc", ""),
                "\n\nSignature template:\n",
                sig,
                f"\n\nCurrent seed (task_assets/problems/{PROBLEM_SUBDIR}/gpt.py):\n```python\n",
                self.seed_code,
                "\n```\n\nInstruction: ",
                diversity_instruction,
                "\n\nRespond with Strategic Thought and one ```python``` block with complete runnable code "
                f"(imports allowed), defining the evolved function like `{self.target_function_name}` "
                "or a versioned variant consistent with the seed.",
            ]
        )


def build_cvrp_pomo_task() -> CvrpPomoTask:
    return CvrpPomoTask()

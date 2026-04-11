"""16 点 min/max 距离比任务：组装种子、提示词、评估。"""

from typing import Any, Dict, List

from tasks.base import EvolutionTask, PromptStrategies
from tasks.min_max_layout_16 import prompts
from tasks.min_max_layout_16.constants import TARGET_FUNCTION_NAME
from tasks.min_max_layout_16.evaluation import run_evaluation
from tasks.min_max_layout_16.seed import SEED_CODE, SEED_THOUGHT


class MinMaxLayout16Task(EvolutionTask):
    id = "min_max_layout_16"
    target_function_name = TARGET_FUNCTION_NAME

    @property
    def seed_code(self) -> str:
        return SEED_CODE.strip()

    @property
    def seed_thought(self) -> str:
        return SEED_THOUGHT

    def evaluate_raw(self, code: str) -> Dict[str, Any]:
        return run_evaluation(code)

    @property
    def prompt_strategies(self) -> PromptStrategies:
        return {
            "modification": prompts.implement_modification_prompt_strategy,
            "exploration": prompts.implement_exploration_prompt_strategy,
            "simplification": prompts.implement_simplification_prompt_strategy,
        }

    @property
    def initial_population_system_prompt(self) -> str:
        return prompts.INITIAL_POPULATION_SYSTEM_PROMPT

    @property
    def diversity_instructions(self) -> List[str]:
        return list(prompts.INITIAL_DIVERSITY_INSTRUCTIONS)

    def format_diversity_user_prompt(self, diversity_instruction: str) -> str:
        fn = self.target_function_name
        return (
            "The current seed individual uses this strategy and code:\n\n"
            f"Strategic Thought: '{self.seed_thought}'\n\n"
            "Code:\n"
            f"{self.seed_code}\n\n"
            f"{diversity_instruction}\n\n"
            "Please produce a new strategic thought and a complete Python constructor "
            f"function named {fn}() (no required arguments). "
            "Ensure the returned code is runnable and returns a numpy array of shape (16, 2)."
        )


def build_min_max_layout_16_task() -> MinMaxLayout16Task:
    return MinMaxLayout16Task()

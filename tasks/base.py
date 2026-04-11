from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

from tasks.task_support.text_utils import ensure_target_function_name

# 个体字典约定：thought, code, fitness（至少含 combined_score 供选择器使用）
Individual = Dict[str, Any]
PromptStrategies = Dict[str, Callable[..., Tuple[str, str]]]
FAILED_COMBINED_SCORE = -1e18


def normalize_standard_fitness(result: Any) -> Dict[str, Any]:
    """将评估器返回结果规范为含 combined_score / eval_time / error，并保留任务特有字段。"""
    normalized: Dict[str, Any] = {
        "combined_score": FAILED_COMBINED_SCORE,
        "eval_time": 0.0,
        "error": None,
    }
    if not isinstance(result, dict):
        normalized["error"] = "Evaluator returned non-dict result."
        return normalized
    if "combined_score" in result:
        try:
            normalized["combined_score"] = float(result.get("combined_score", 0.0))
        except (TypeError, ValueError):
            normalized["combined_score"] = FAILED_COMBINED_SCORE
    if "eval_time" in result:
        try:
            normalized["eval_time"] = float(result.get("eval_time", 0.0))
        except (TypeError, ValueError):
            normalized["eval_time"] = 0.0
    if result.get("error") is not None:
        normalized["error"] = str(result.get("error"))
        normalized["combined_score"] = FAILED_COMBINED_SCORE
    elif normalized["combined_score"] == 0.0:
        alt = result.get("message") or result.get("msg") or result.get("exception")
        if alt:
            normalized["error"] = str(alt)
            normalized["combined_score"] = FAILED_COMBINED_SCORE
    for key, value in result.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def fitness_has_error(fitness: Any) -> bool:
    if not isinstance(fitness, dict):
        return True
    err = fitness.get("error")
    return err is not None and str(err).strip() != ""


def individual_is_valid(individual: Any) -> bool:
    if not isinstance(individual, dict):
        return False
    return not fitness_has_error(individual.get("fitness", {}))


class EvolutionTask(ABC):
    """演化框架与具体优化问题之间的契约。"""

    id: str
    target_function_name: str

    @property
    @abstractmethod
    def seed_code(self) -> str:
        ...

    @property
    @abstractmethod
    def seed_thought(self) -> str:
        ...

    @abstractmethod
    def evaluate_raw(self, code: str) -> Dict[str, Any]:
        """运行任务内置评估，返回原始 dict（可含 error）。"""

    def evaluate(self, code: str) -> Dict[str, Any]:
        return normalize_standard_fitness(self.evaluate_raw(code))

    def run_full_test(self, code: str, *, mode: str = "test") -> Dict[str, Any]:
        raise NotImplementedError(f"Task `{self.id}` does not support full test mode.")

    def validate_syntax(self, code: str) -> None:
        from implement_llm_interaction_module.implement_response_parser import validate_constructor_syntax

        validate_constructor_syntax(code, function_name=self.target_function_name)

    def extract_thought_and_code(self, llm_response: str) -> Dict[str, str]:
        from implement_llm_interaction_module.implement_response_parser import extract_thought_and_code_sections

        sections = extract_thought_and_code_sections(
            llm_response, target_function_name=self.target_function_name
        )
        code = sections.get("code", "")
        if code:
            sections["code"] = ensure_target_function_name(code, self.target_function_name)
        return sections

    @property
    @abstractmethod
    def prompt_strategies(self) -> PromptStrategies:
        """modification / exploration / simplification -> (system, user) 生成函数。"""

    @property
    @abstractmethod
    def initial_population_system_prompt(self) -> str:
        """用于初始种群 LLM 生成的 system prompt。"""

    @property
    def diversity_instructions(self) -> List[str]:
        """初始多样化指令列表（可被子类覆盖）。"""
        return [
            "Generate a candidate that uses a distinct algorithmic or probabilistic approach.",
            "Please invent a new strategy differing from the seed.",
            "Try a novel method for the implementation.",
            "Change both the code and the strategic thought for more diversity.",
        ]

    def format_diversity_user_prompt(self, diversity_instruction: str) -> str:
        """基于种子与一条多样化指令，生成调用 LLM 时的 user prompt（任务可覆盖以加入 I/O 约定等）。"""
        return (
            "The current seed individual uses this strategy and code:\n\n"
            f"Strategic Thought: '{self.seed_thought}'\n\n"
            "Code:\n"
            f"{self.seed_code}\n\n"
            f"{diversity_instruction}\n\n"
            "Please produce a new strategic thought and complete runnable Python code "
            f"that defines the task entry function `{self.target_function_name}`. "
            "Keep the interface consistent with the task's seed/signature expectations."
        )

    def fallback_seed_code(self, variant_index: int) -> str:
        """在 LLM 生成不可用时，为初始化种群提供后备代码。任务可覆盖以生成真正变体。"""
        if not isinstance(variant_index, int) or variant_index < 0:
            raise ValueError("variant_index must be a non-negative integer")
        return self.seed_code

    def fallback_seed_thought(self, variant_index: int) -> str:
        """在 LLM 生成不可用时，为初始化种群提供后备策略说明。"""
        if not isinstance(variant_index, int) or variant_index < 0:
            raise ValueError("variant_index must be a non-negative integer")
        return f"Fallback seed reuse #{variant_index + 1} for task `{self.id}`."

    def build_fallback_individual(self, variant_index: int) -> Individual:
        """构造一个后备个体；默认复用种子，任务可覆盖以生成更丰富的本地初始化。"""
        code = self.fallback_seed_code(variant_index)
        thought = self.fallback_seed_thought(variant_index)
        self.validate_syntax(code)
        return {
            "thought": thought,
            "code": code,
            "fitness": self.evaluate(code),
        }

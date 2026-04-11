"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def heuristics_v1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:\n    return prize / np.sum(weight, axis=1)',
    'func_signature': 'def heuristics_v{version}(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:',
    'func_desc': 'Suppose `n` indicates the scale of the problem, and `m` is the dimension of weights each item has. The constraint of each dimension is fixed to 1. The `heuristics` function takes as input a `prize` of shape (n,), a `weight` of shape (n, m), and returns `heuristics` of shape (n,). `heuristics[i]` indicates how promising it is to include item i in the solution.',
    'external_knowledge': '- Try combining various factors to determine how promising it is to select an item.\n- Try sparsifying the heuristics by setting unpromising elements to zero.',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': "def heuristics_v1(distance_matrix: np.ndarray) -> np.ndarray:\n    # It's bad to include long edges in the solution\n    return distance_matrix",
    'func_signature': 'def heuristics_v{version}(distance_matrix: np.ndarray) -> np.ndarray:',
    'func_desc': 'The `heuristics` function takes as input a distance matrix, and returns prior indicators of how bad it is to include each edge in a solution. The return is of the same shape as the input.',
    'external_knowledge': '',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

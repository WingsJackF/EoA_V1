"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def heuristics_v1(demand: np.ndarray, capacity: int) -> np.ndarray:\n    return np.tile(demand/demand.max(), (demand.shape[0], 1))',
    'func_signature': 'def heuristics_v{version}(demand: np.ndarray, capacity: int) -> np.ndarray:',
    'func_desc': 'Suppose `n` represents the number of items in the problem. The heuristics function takes as input a `demand` array of shape (n,) and an integer as the capacity of every bin, and it returns a `heuristics` array of shape (n,n). `heuristics[i][j]` indicates how promising it is to put item i and item j in the same bin.',
    'external_knowledge': '- Try combining various factors to determine how promising it is to select an edge.\n- Try sparsifying the matrix by setting unpromising elements to zero.',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

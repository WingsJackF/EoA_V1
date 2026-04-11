"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def heuristics_v1(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:\n    return prize[np.newaxis, :] / distance',
    'func_signature': 'def heuristics_v{version}(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:',
    'func_desc': 'Suppose `n` represents the number of nodes in the problem, with the depot being the first node. The `heuristics` function takes as input a `prize` array of shape (n,), a `distance` matrix of shape (n,n), and a `max_len` float which is the constraint to total travel distance, and it returns `heuristics` of shape (n, n), where `heuristics[i][j]` indicates the promise of including the edge from node #i to node #j in the solution.',
    'external_knowledge': '- Try combining various factors to determine how promising it is to select an edge.\n- Try sparsifying the matrix by setting unpromising elements to zero.',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

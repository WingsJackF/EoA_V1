"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def crossover_v1(parents: np.ndarray, n_pop: int) -> np.ndarray:\n    n_parents, n_decap = parents.shape\n\n    # Split genomes into two halves\n    left_halves = parents[:, :n_decap // 2]\n    right_halves = parents[:, n_decap // 2:]\n\n    # Create parent pairs\n    parents_idx = np.stack([np.random.choice(range(n_parents), 2, replace=False) for _ in range(n_pop)])\n    parents_left = left_halves[parents_idx[:, 0]]\n    parents_right = right_halves[parents_idx[:, 1]]\n\n    # Create offspring\n    offspring = np.concatenate([parents_left, parents_right], axis=1)\n    return offspring',
    'func_signature': 'def crossover_v{version}(parents: np.ndarray, n_pop: int) -> np.ndarray:',
    'func_desc': 'The `crossover` function takes as input a 2D NumPy array parents and an integer n_pop. The function performs a genetic crossover operation on parents to generate n_pop offspring. Use vectorized implementation if possible.',
    'external_knowledge': '',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

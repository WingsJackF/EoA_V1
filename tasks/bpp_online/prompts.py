"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def priority_v1(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:\n  """Best known heuristics."""\n  max_bin_cap = max(bins_remain_cap)\n  score = (bins_remain_cap - max_bin_cap)**2 / item + bins_remain_cap**2 / (item**2)\n  score += bins_remain_cap**2 / item**3\n  score[bins_remain_cap > item] = -score[bins_remain_cap > item]\n  score[1:] -= score[:-1]\n  return score',
    'func_signature': 'def priority_v{version}(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:',
    'func_desc': 'The priority function takes as input an item and an array of bins_remain_cap (containing the remaining capacity of each bin) and returns a priority score for each bin. The bin with the highest priority score will be selected for the item.',
    'external_knowledge': '',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

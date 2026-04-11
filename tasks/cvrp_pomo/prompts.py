"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def heuristics_v1(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:\n    """A trivial implementation to improve upon."""\n    return torch.zeros_like(distance_matrix)',
    'func_signature': 'def heuristics_v{version}(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:',
    'func_desc': 'The `heuristics` function takes as input a distance matrix (shape: n by n) and a vector of customer demands (shape: n), where the depot node is indexed by 0 and the customer demands are normalized by the total vehicle capacity. It returns prior indicators of how promising it is to include each edge in a solution. The return is of the same shape as the distance matrix. The heuristics should contain negative values for undesirable edges and positive values for promising ones. Use efficient vectorized implementations.',
    'external_knowledge': '',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

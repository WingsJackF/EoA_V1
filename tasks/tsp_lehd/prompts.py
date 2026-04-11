"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def heuristics_v1(distance_matrix: torch.Tensor) -> torch.Tensor:\n    """\n    heu_ij = - log(dis_ij) if j is the topK nearest neighbor of i, else - dis_ij\n    """\n    distance_matrix[distance_matrix == 0] = 1e5\n    K = 100\n    # Compute top-k nearest neighbors (smallest distances)\n    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)\n    heu = -distance_matrix.clone()\n    # Create a mask where topk indices are True and others are False\n    topk_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)\n    topk_mask.scatter_(1, indices, True)\n    # Apply -log(d_ij) only to the top-k elements\n    heu[topk_mask] = -torch.log(distance_matrix[topk_mask])\n    return heu',
    'func_signature': 'def heuristics_v{version}(distance_matrix: torch.Tensor) -> torch.Tensor:',
    'func_desc': 'The `heuristics` function takes as input a distance matrix and returns prior indicators of how bad it is to include each edge in a solution. The return is of the same shape as the input. The heuristics should contain negative values for undesirable edges and positive values for promising ones. Use efficient vectorized implementations.',
    'external_knowledge': '',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

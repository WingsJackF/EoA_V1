"""内联提示词资产。"""

from tasks.task_support.prompts_builder import (
    build_prompt_strategies,
    diversity_lines_from_assets,
    initial_system_prompt,
)

from .constants import TASK_LABEL

PROMPT_ASSETS: dict[str, str] = {
    'seed_func': 'def select_next_node_v1(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:\n    """Select the next node to visit from the unvisited nodes."""\n    threshold = 0.7\n    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1\n    scores = {}\n    for node in unvisited_nodes:\n        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]\n        average_distance_to_unvisited = np.mean(all_distances)\n        std_dev_distance_to_unvisited = np.std(all_distances)\n        score = c1 * distance_matrix[current_node][node] - c2 * average_distance_to_unvisited + c3 * std_dev_distance_to_unvisited - c4 * distance_matrix[destination_node][node]\n        scores[node] = score\n    next_node = min(scores, key=scores.get)\n    return next_node',
    'func_signature': 'def select_next_node_v{version}(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:',
    'func_desc': 'The select_next_node function takes as input the current node, the destination_node, a set of unvisited nodes, and a distance matrix, and returns the next node to visit.',
    'external_knowledge': '- Try look-ahead mechanisms.',
}

PROMPT_STRATEGIES = build_prompt_strategies(PROMPT_ASSETS)
INITIAL_POPULATION_SYSTEM_PROMPT = initial_system_prompt(PROMPT_ASSETS, TASK_LABEL)
DIVERSITY_INSTRUCTIONS = diversity_lines_from_assets(PROMPT_ASSETS)

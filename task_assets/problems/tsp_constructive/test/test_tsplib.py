import numpy as np
from scipy.spatial import distance_matrix
from copy import copy
from tqdm import tqdm

try:
    from gpt import select_next_node_v2 as select_next_node
except Exception:
    try:
        from gpt import select_next_node
    except Exception:
        select_next_node = None

# Datasets and baseline results are drawn from Duflo et al. (2019). A GP Hyper-Heuristic Approach for Generating TSP Heuristics.

names = ["ts225", "rat99", "rl1889", "u1817", "d1655", "bier127", "lin318", "eil51", "d493", "kroB100", "kroC100", "ch130", "pr299", "fl417", "d657", "kroA150", "fl1577", "u724", "pr264", "pr226", "pr439"]

opt = {
    'ts225': 126643,
    'rat99': 1211,
    'rl1889': 316536,
    'u1817': 57201,
    'd1655': 62128,
    'bier127': 118282,
    'lin318': 42029,
    'eil51': 426,
    'd493': 35002,
    'kroB100': 22141,
    'kroC100': 20749,
    'ch130': 6110,
    'pr299': 48191,
    'fl417': 11861,
    'd657': 48912,
    'kroA150': 26524,
    'fl1577': 22249,
    'u724': 41910,
    'pr264': 49135,
    'pr226': 80369,
    'pr439': 107217
 }


def select_next_node_candidate(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    # """Select the next node to visit from the unvisited nodes."""
    # weights = {'distance_to_current': 0.4, 
    #            'average_distance_to_unvisited': 0.25, 
    #            'std_dev_distance_to_unvisited': 0.25, 
    #            'distance_to_destination': 0.1}
    # scores = {}
    # for node in unvisited_nodes:
    #     future_distances = [distance_matrix[node, i] for i in unvisited_nodes if i != node]
    #     if future_distances:
    #         average_distance_to_unvisited = sum(future_distances) / len(future_distances)
    #         std_dev_distance_to_unvisited = (sum((x - average_distance_to_unvisited) ** 2 for x in future_distances) / len(future_distances)) ** 0.5
    #     else:
    #         average_distance_to_unvisited = std_dev_distance_to_unvisited = 0
    #     score = (weights['distance_to_current'] * distance_matrix[current_node, node] -
    #              weights['average_distance_to_unvisited'] * average_distance_to_unvisited +
    #              weights['std_dev_distance_to_unvisited'] * std_dev_distance_to_unvisited -
    #              weights['distance_to_destination'] * distance_matrix[destination_node, node])
    #     scores[node] = score
    # next_node = min(scores, key=scores.get)
    # return next_node
    """Select the next node to visit from the unvisited nodes."""
    threshold = 0.7
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)
        score = c1 * distance_matrix[current_node][node] - c2 * average_distance_to_unvisited + c3 * std_dev_distance_to_unvisited - c4 * distance_matrix[destination_node][node]
        scores[node] = score
    next_node = min(scores, key=scores.get)
    return next_node


def _resolve_selector():
    return select_next_node if callable(select_next_node) else select_next_node_candidate


def eval_heuristic(node_positions: np.ndarray, start_node: int, *, show_progress: bool = False) -> float:
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    selector = _resolve_selector()
    # run the heuristic
    for _ in tqdm(range(problem_size - 1), disable=not show_progress):
        next_node = selector(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj


def _load_tsplib_instance(name: str) -> tuple[np.ndarray, float]:
    with open(f'tsplib/{name}.tsp') as f:
        lines = f.readlines()

    data = lines[6:-1]
    data = [x.strip().split() for x in data]
    data = [[float(x) for x in row[1:]] for row in data]
    data = np.array(data)
    scale = max(np.max(data, axis=0) - np.min(data, axis=0))
    data = (data - np.min(data, axis=0)) / scale
    return data, scale


def run_tsplib_test(*, show_progress: bool = False) -> dict:
    results = {}
    opt_gaps = []

    for name in names:
        data, scale = _load_tsplib_instance(name)
        objs = []
        for start_node in range(3):
            obj = eval_heuristic(data, start_node, show_progress=show_progress) * scale
            objs.append(obj)
        mean = float(np.mean(objs))
        std = float(np.std(objs))
        gap = float((mean - opt[name]) / opt[name] * 100)
        results[name] = {
            "objective_mean": mean,
            "objective_std": std,
            "optimal_value": float(opt[name]),
            "gap_percent": gap,
        }
        opt_gaps.append(gap)

    return {
        "instances": results,
        "mean_gap_percent": float(np.mean(opt_gaps)),
        "error": None,
    }


if __name__ == '__main__':
    payload = run_tsplib_test(show_progress=True)
    for name in names:
        stats = payload["instances"][name]
        print(name)
        print(f"\tObjective: {stats['objective_mean']}+-{stats['objective_std']}")
        print(f"\tOpt. Gap: {stats['gap_percent']}%")
        print()
    print(f"Mean Gap: {payload['mean_gap_percent']}%")

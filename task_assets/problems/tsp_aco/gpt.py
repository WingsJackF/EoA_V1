import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    eps = 1e-10
    dist = np.where(distance_matrix == np.inf, np.inf, distance_matrix) + eps

    # Heuristic 1: Inverse distance – prioritize shorter edges
    inv_dist = 1.0 / dist
    inv_dist = np.where(np.isinf(inv_dist), 0, inv_dist)

    # Heuristic 2: Node centrality – favor edges between low-degree nodes
    degrees = np.sum(dist < np.inf, axis=1)
    node_centrality = 1.0 / (degrees + eps)
    centrality_edge_score = np.outer(node_centrality, node_centrality)

    # Heuristic 3: Local diversity – prefer connecting to less connected neighbors
    k = min(5, n)
    sorted_neighbors = np.argsort(dist, axis=1)
    diversity_score = np.zeros_like(dist)

    for i in range(n):
        neighbors = sorted_neighbors[i, :k]
        proximity_weight = 1.0 / (np.arange(1, k + 1) + eps)
        connectivity_penalty = 1.0 / (degrees[neighbors] + eps)
        diversity_score[i, neighbors] = proximity_weight * connectivity_penalty

    # Heuristic 4: Symmetry-aware preference – promote consistent bidirectional edges
    connectivity = np.where(dist < np.inf, 1.0, 0.0)
    symmetry_score = (connectivity + connectivity.T) / 2

    # Heuristic 5: Balance – reduce hub dominance using geometric mean
    hub_balance = np.sqrt(node_centrality[:, None] * node_centrality[None, :])
    balance_score = 1.0 / (hub_balance + eps)

    # Adaptive weighting based on graph density
    avg_degree = np.mean(degrees)
    density_factor = np.clip(avg_degree / n, 0.05, 0.8)
    alpha = 1.8 * (1 - density_factor)  # Higher centrality in sparse graphs
    beta = 1.4 * density_factor         # Higher diversity in dense graphs
    gamma = 1.1 * (1 - 0.4 * density_factor)  # Less symmetry emphasis in dense
    delta = 2.2 * density_factor       # More balance emphasis in dense

    # Unified geometric fusion with entropy-inspired shaping
    combined = (
        inv_dist ** 1.35 *
        (centrality_edge_score + eps) ** alpha *
        (diversity_score + eps) ** beta *
        (symmetry_score + eps) ** gamma *
        (balance_score + eps) ** delta
    )

    # Entropy-guided sparsification with top-k edges per node
    k = max(2, int(0.25 * n))
    top_edges_per_node = np.argsort(combined, axis=1)[:, -k:]

    # Build sparse heuristic matrix
    result = np.zeros_like(combined)
    for i in range(n):
        result[i, top_edges_per_node[i]] = combined[i, top_edges_per_node[i]]

    # Normalize per node to maintain relative edge quality
    row_norms = np.sum(result, axis=1, keepdims=True)
    result = np.where(row_norms > 0, result / (row_norms + eps), result)

    # Enforce symmetry (undirected graph)
    result = (result + result.T) / 2

    # Remove self-loops
    np.fill_diagonal(result, 0)

    return result

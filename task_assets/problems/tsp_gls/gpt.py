import numpy as np
import numpy as np
from typing import Tuple

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Enhanced heuristic focusing on central nodes and edge uniqueness.
    Returns penalty matrix where higher values indicate worse edges for TSP.
    """
    n = len(distance_matrix)
    
    if n <= 2:
        return np.zeros_like(distance_matrix) if n > 0 else distance_matrix
    
    # 1. STRONG centrality penalty - edges connecting central nodes get high penalty
    # Central nodes are those with small average distance to other nodes
    avg_distances = np.sum(distance_matrix, axis=1) / (n - 1)
    
    # Invert so smaller average distance = higher centrality score
    min_avg, max_avg = np.min(avg_distances), np.max(avg_distances)
    if max_avg > min_avg:
        centrality = 1.0 - (avg_distances - min_avg) / (max_avg - min_avg)
    else:
        centrality = np.ones(n)
    
    # Amplify centrality differences using power scaling
    centrality = centrality ** 2
    
    # Create centrality penalty matrix
    # Using outer product for efficient computation
    centrality_matrix = np.outer(centrality, centrality)
    
    # 2. STRONG edge uniqueness penalty - edges with few alternatives get high penalty
    # Sort distances to find k-nearest neighbors for each node
    k = min(10, n - 1)
    
    # Get sorted indices for each row (excluding self)
    sorted_indices = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
    
    # Create neighbor count matrix
    neighbor_count = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        # Mark k-nearest neighbors
        neighbor_count[i, sorted_indices[i]] = 1
    
    # Symmetrize - an edge is "common" if it's in k-nearest of either endpoint
    common_edges = np.maximum(neighbor_count, neighbor_count.T)
    
    # Uniqueness penalty: edges NOT in k-nearest neighbors get higher penalty
    uniqueness_penalty = 1.0 - common_edges.astype(float)
    
    # 3. Edge length component (normalized)
    max_dist = np.max(distance_matrix)
    if max_dist > 0:
        length_penalty = distance_matrix / max_dist
    else:
        length_penalty = np.zeros_like(distance_matrix)
    
    # 4. Combined penalty with emphasis on centrality and uniqueness
    # Weights can be adjusted based on problem characteristics
    penalty_matrix = (
        0.4 * centrality_matrix +    # Strong focus on central nodes
        0.4 * uniqueness_penalty +   # Strong focus on unique edges  
        0.2 * length_penalty         # Moderate focus on edge length
    )
    
    # Ensure symmetry
    penalty_matrix = (penalty_matrix + penalty_matrix.T) / 2
    
    # Zero diagonal
    np.fill_diagonal(penalty_matrix, 0)
    
    # Normalize to [0, 1] range
    max_penalty = np.max(penalty_matrix)
    if max_penalty > 0:
        penalty_matrix = penalty_matrix / max_penalty
    
    return penalty_matrix

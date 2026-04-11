import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    """
    Enhanced CVRP heuristic with exponential savings, angular clustering, 
    dynamic capacity thresholds, and adaptive sparsification for stochastic sampling.
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros_like(distance_matrix)
    
    heuristics = np.zeros((n, n))
    
    # 1. Exponential distance scoring with depot normalization
    depot_coord = coordinates[0]
    depot_distances = np.linalg.norm(coordinates - depot_coord, axis=1)
    mean_depot_dist = np.mean(depot_distances[1:]) if len(depot_distances) > 1 else 1.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        distance_inv = 1.0 / (distance_matrix + np.eye(n) * 1e-10)
        distance_inv[np.isinf(distance_inv)] = 0
        distance_inv[np.isnan(distance_inv)] = 0
    
    # Exponential decay scaled by depot distance
    distance_decay = np.exp(-distance_matrix / (0.5 * mean_depot_dist + 1e-10))
    distance_inv = distance_inv * distance_decay
    
    # 2. Angular clustering around depot
    angular_scores = np.ones((n, n))
    for i in range(1, n):
        vec_i = coordinates[i] - depot_coord
        norm_i = np.linalg.norm(vec_i)
        if norm_i > 1e-10:
            vec_i = vec_i / norm_i
            for j in range(i + 1, n):
                vec_j = coordinates[j] - depot_coord
                norm_j = np.linalg.norm(vec_j)
                if norm_j > 1e-10:
                    vec_j = vec_j / norm_j
                    cos_angle = np.dot(vec_i, vec_j)
                    # Exponential reward for similar angles (clustered nodes)
                    angular_scores[i, j] = angular_scores[j, i] = np.exp(3.0 * (cos_angle - 0.7))
    
    # 3. Exponential savings heuristic with dynamic scaling
    savings_matrix = np.ones((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            savings = depot_distances[i] + depot_distances[j] - distance_matrix[i, j]
            # Normalize by mean depot distance and apply exponential transformation
            normalized_savings = savings / (mean_depot_dist + 1e-10)
            savings_matrix[i, j] = savings_matrix[j, i] = np.exp(3.2 * max(normalized_savings, 0))
    
    # Depot connections: reward based on demand-to-capacity ratio
    for i in range(1, n):
        demand_ratio = demands[i] / capacity
        savings_matrix[0, i] = savings_matrix[i, 0] = np.exp(2.5 * demand_ratio)
    
    # 4. Capacity feasibility with exponential penalties/rewards
    capacity_factor = np.ones((n, n))
    avg_demand = np.mean(demands[1:]) if len(demands) > 1 else capacity
    capacity_threshold = min(0.75 * capacity, max(0.6 * capacity, 1.8 * avg_demand))
    
    for i in range(1, n):
        for j in range(i + 1, n):
            demand_sum = demands[i] + demands[j]
            if demand_sum <= capacity:
                # Exponential reward for fitting well within capacity
                fill_ratio = demand_sum / capacity
                capacity_factor[i, j] = capacity_factor[j, i] = np.exp(4.0 * fill_ratio)
            else:
                # Exponential penalty for exceeding capacity
                excess_ratio = (demand_sum - capacity) / capacity
                capacity_factor[i, j] = capacity_factor[j, i] = 0.001 * np.exp(-8.0 * excess_ratio)
    
    # Depot connections: dynamic boost based on demand threshold
    for i in range(1, n):
        demand_ratio = demands[i] / capacity
        if demands[i] > capacity_threshold:
            depot_boost = 1.0 + 5.0 * np.exp(3.0 * (demand_ratio - 0.85))
        else:
            depot_boost = 1.0 + 2.5 * np.exp(2.0 * (demand_ratio - 0.65))
        capacity_factor[0, i] = capacity_factor[i, 0] = depot_boost
    
    # 5. Demand complementarity via weighted geometric mean
    complementarity = np.ones((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            demand_i = demands[i] / capacity
            demand_j = demands[j] / capacity
            
            # Weighted geometric mean with exponential transformation
            weight_i = demand_i / (demand_i + demand_j + 1e-10)
            weight_j = demand_j / (demand_i + demand_j + 1e-10)
            geo_mean = (demand_i ** weight_i) * (demand_j ** weight_j)
            complementarity[i, j] = complementarity[j, i] = np.exp(2.5 * geo_mean)
    
    # 6. Combine factors with multiplicative scoring
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            base_score = distance_inv[i, j]
            
            if i == 0 or j == 0:
                # Depot connections: emphasize capacity and savings
                customer = j if i == 0 else i
                demand_ratio = demands[customer] / capacity
                dist_factor = np.exp(-depot_distances[customer] / (1.2 * mean_depot_dist))
                demand_factor = 1.0 + 4.5 * np.exp(3.0 * (demand_ratio - 0.8))
                
                heuristics[i, j] = (base_score * demand_factor * dist_factor * 
                                   savings_matrix[i, j] * capacity_factor[i, j])
            else:
                # Customer-to-customer: weighted combination
                capacity_weight = 1.6
                angular_weight = 1.3
                complementarity_weight = 1.2
                savings_weight = 1.4
                
                heuristics[i, j] = (base_score *
                                   capacity_factor[i, j] ** capacity_weight *
                                   angular_scores[i, j] ** angular_weight *
                                   complementarity[i, j] ** complementarity_weight *
                                   savings_matrix[i, j] ** savings_weight)
    
    # 7. Adaptive sparsification with dynamic threshold
    # Identify high-demand nodes for special treatment
    high_demand_threshold = max(0.8 * capacity, 2.0 * avg_demand)
    high_demand_nodes = np.where(demands > high_demand_threshold)[0]
    
    # Dynamic sparsity based on problem characteristics
    base_sparsity = max(10, min(n // 2, 25))
    demand_density = np.mean(demands[1:] > capacity_threshold) if len(demands) > 1 else 0.5
    k_sparse = int(base_sparsity * (1 + demand_density))
    
    # Row-wise top-k selection with symmetry preservation
    row_masks = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        row_scores = heuristics[i].copy()
        row_scores[i] = -np.inf
        
        # Keep connections to depot for high-demand nodes
        if i == 0:
            row_scores[high_demand_nodes] = np.inf
        elif i in high_demand_nodes:
            row_scores[0] = np.inf
        
        # Find top-k scores for this row
        if np.sum(row_scores > 0) > k_sparse:
            top_k_indices = np.argpartition(row_scores, -k_sparse)[-k_sparse:]
            threshold = np.min(row_scores[top_k_indices])
            row_masks[i] = row_scores >= threshold
        else:
            row_masks[i] = row_scores > 0
    
    # Ensure symmetry
    symmetric_mask = np.logical_or(row_masks, row_masks.T)
    
    # Force connections for high-demand nodes to depot
    for node in high_demand_nodes:
        if node < n:
            symmetric_mask[0, node] = symmetric_mask[node, 0] = True
    
    # Apply sparsification
    heuristics_sparse = np.zeros((n, n))
    heuristics_sparse[symmetric_mask] = heuristics[symmetric_mask]
    
    # 8. Normalize to [0, 1] range
    max_val = np.max(heuristics_sparse)
    if max_val > 0:
        heuristics_sparse = heuristics_sparse / max_val
    
    # Zero diagonal
    np.fill_diagonal(heuristics_sparse, 0)
    
    return heuristics_sparse

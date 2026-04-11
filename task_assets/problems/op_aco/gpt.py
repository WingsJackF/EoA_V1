import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    n = len(prize)
    
    # Create safe distance matrix to avoid division by zero
    safe_dist = distance.copy()
    safe_dist[safe_dist == 0] = 1e-8
    
    # 1. Core prize-to-distance ratio (normalized)
    prize_to_dist = prize / safe_dist
    ptd_max = prize_to_dist.max()
    if ptd_max > 0:
        prize_to_dist = prize_to_dist / ptd_max
    
    # 2. Budget feasibility factor (exponential decay)
    dist_ratio = distance / maxlen
    budget_factor = np.exp(-3 * dist_ratio)
    
    # 3. Simple prize diffusion via normalized adjacency
    # Use inverse distance as connectivity measure
    with np.errstate(divide='ignore'):
        connectivity = 1.0 / safe_dist
    np.fill_diagonal(connectivity, 0)
    
    # Row-normalize to get weights
    row_sums = connectivity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = connectivity / row_sums
    
    # Compute diffused prize scores
    diffused_prize = weights @ prize
    dp_max = diffused_prize.max()
    if dp_max > 0:
        diffused_prize = diffused_prize / dp_max
    
    # 4. Combined score with clear weighting
    combined = (prize_to_dist * 
                budget_factor * 
                (0.5 + 0.5 * diffused_prize[:, np.newaxis]))
    
    # 5. Depot handling: prioritize leaving depot, encourage returning
    combined[0, :] *= 1.3  # From depot bonus
    combined[:, 0] *= 1.2  # To depot bonus
    
    # 6. Adaptive sparsification - keep more edges for depot
    k_base = max(3, n // 10)
    heur = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            # Depot: keep more connections
            k = min(n, k_base * 2)
        else:
            k = k_base
        
        # Get top-k edges
        sorted_indices = np.argsort(combined[i])[::-1]
        top_k_indices = sorted_indices[:k]
        heur[i, top_k_indices] = combined[i, top_k_indices]
    
    # 7. Final normalization
    h_max = heur.max()
    if h_max > 0:
        heur = heur / h_max
    
    # Zero diagonal
    np.fill_diagonal(heur, 0)
    
    return heur

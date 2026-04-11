import numpy as np
import numpy as np

def heuristics_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
    """
    Enhanced heuristics for Bin Packing Problem focusing on:
    1. EXACT FITS: Highest priority for pairs that exactly fill bins
    2. MINIMAL WASTE: Secondary priority for complementary pairs
    3. SYMMETRIC SPARSIFICATION: Ensures consistent pairwise relationships
    4. LARGE ITEM PRIORITIZATION: Special handling for hard-to-pack items
    """
    
    n = len(demand)
    heuristics = np.zeros((n, n))
    
    # Pre-compute pairwise sums
    demand_sum = demand[:, None] + demand
    
    # 1. FEASIBILITY CHECK - Absolute requirement
    feasible_mask = demand_sum <= capacity
    infeasible_penalty = -100.0  # Very strong penalty
    
    # 2. EXACT FIT BONUS - Highest priority
    exact_fit = (demand_sum == capacity)
    exact_fit_score = np.where(exact_fit, 10.0, 0.0)
    
    # 3. WASTE MINIMIZATION - Complementary sizes
    waste = capacity - demand_sum
    waste_score = np.where(feasible_mask, 
                           np.maximum(0, 1.0 - (waste / capacity) ** 2),
                           0.0)
    
    # 4. LARGE ITEM PRIORITIZATION
    large_item_threshold = capacity * 0.8
    is_large = demand > large_item_threshold
    large_item_bonus = np.where(is_large[:, None] & is_large & feasible_mask,
                                2.0,  # Bonus for pairing two large items
                                0.0)
    
    # 5. SYMMETRIC COMPLEMENTARITY
    # How well does j complement i's leftover space?
    leftover_space = capacity - demand
    complement_gap = np.abs(leftover_space[:, None] - demand)
    complement_score = np.where(feasible_mask,
                                1.0 - (complement_gap / capacity),
                                0.0)
    
    # Combine all factors with prioritization weights
    heuristics = (
        4.0 * exact_fit_score +      # Highest priority: exact fits
        2.0 * waste_score +          # High priority: minimal waste
        1.0 * large_item_bonus +     # Medium priority: large items
        0.5 * complement_score       # Low priority: general complementarity
    )
    
    # Apply feasibility penalty
    heuristics = np.where(feasible_mask, heuristics, infeasible_penalty)
    
    # Zero diagonal
    np.fill_diagonal(heuristics, 0.0)
    
    # SYMMETRIC SPARSIFICATION
    # Keep only top connections while maintaining symmetry
    k = max(2, int(np.sqrt(n)))  # Number of connections to keep per item
    
    for i in range(n):
        # Get valid (feasible) connections for this item
        valid_mask = feasible_mask[i] & (np.arange(n) != i)
        if not np.any(valid_mask):
            continue
            
        # Get row values for valid connections
        row_values = heuristics[i].copy()
        row_values[~valid_mask] = -np.inf
        
        # Get indices of top k values
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > k:
            # Get top k indices among valid connections
            top_k_indices = valid_indices[np.argsort(row_values[valid_indices])[-k:]]
            
            # Zero out non-top connections
            non_top_mask = np.ones(n, dtype=bool)
            non_top_mask[top_k_indices] = False
            non_top_mask[i] = False  # Keep diagonal as zero
            heuristics[i, non_top_mask] = 0.0
    
    # Ensure symmetry: if i->j is kept, j->i should also be kept
    symmetric_mask = (heuristics > 0) & (heuristics.T > 0)
    heuristics = np.where(symmetric_mask, heuristics, 0.0)
    
    # Final cleanup: zero out any remaining negative values
    heuristics = np.maximum(heuristics, 0.0)
    
    # Normalize to [0, 1] range for consistency
    max_val = np.max(heuristics)
    if max_val > 0:
        heuristics = heuristics / max_val
    
    return heuristics

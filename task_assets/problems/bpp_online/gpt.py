import numpy as np
import numpy as np

def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    # Transform bins remain capacity with logarithmic scaling to accentuate differences in small capacities
    log_capacity = np.log(1 + bins_remain_cap) + 1e-8  # Adds small epsilon to avoid log(0)
    
    # Penalty for bins where item won't fit
    can_fit = bins_remain_cap >= item
    fit_score = np.where(can_fit, 0, -np.inf)
    
    # Encourage homogeneity among bins by favoring bins closer to average residual capacity
    avg_residual = np.mean(bins_remain_cap) if len(bins_remain_cap) > 0 else 0
    favor_locality = np.exp(-10 * (bins_remain_cap - avg_residual)**2 / (avg_residual + 1e-5))
    
    # Increasing preference for bins with remaining space similar to item size
    favor_item_proportion = np.exp(-np.abs(bins_remain_cap - item))
    
    # Favor bins with very high capacity, possibly leaving room for future large items (with a diminishing factor)
    encourage_fullness = np.power(bins_remain_cap / (1 + item), 0.8)

    # Weighted composite score
    score = (
        10 * fit_score +
        8 * favor_item_proportion +
        7 * favor_locality +
        6 * encourage_fullness +
        5 * log_capacity +
        4 * np.where(can_fit, bins_remain_cap, 0)
    )
    
    # Add noise to break ties and improve exploration slightly
    np.random.seed(hash(str(item)) % 100 + int(time.time() * 1000) % 100)  # Syntactic: hash-based seed to impact randomness
    noise = np.random.normal(0, 0.2, score.shape)
    
    return score + noise

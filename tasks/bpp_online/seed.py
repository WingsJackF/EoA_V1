"""
内联种子代码。
"""

SEED_CODE = "import numpy as np\nimport numpy as np\nimport time\n\ndef priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:\n    # Transform bins remain capacity with logarithmic scaling to accentuate differences in small capacities\n    log_capacity = np.log(1 + bins_remain_cap) + 1e-8  # Adds small epsilon to avoid log(0)\n    \n    # Penalty for bins where item won't fit\n    can_fit = bins_remain_cap >= item\n    fit_score = np.where(can_fit, 0, -np.inf)\n    \n    # Encourage homogeneity among bins by favoring bins closer to average residual capacity\n    avg_residual = np.mean(bins_remain_cap) if len(bins_remain_cap) > 0 else 0\n    favor_locality = np.exp(-10 * (bins_remain_cap - avg_residual)**2 / (avg_residual + 1e-5))\n    \n    # Increasing preference for bins with remaining space similar to item size\n    favor_item_proportion = np.exp(-np.abs(bins_remain_cap - item))\n    \n    # Favor bins with very high capacity, possibly leaving room for future large items (with a diminishing factor)\n    encourage_fullness = np.power(bins_remain_cap / (1 + item), 0.8)\n\n    # Weighted composite score\n    score = (\n        10 * fit_score +\n        8 * favor_item_proportion +\n        7 * favor_locality +\n        6 * encourage_fullness +\n        5 * log_capacity +\n        4 * np.where(can_fit, bins_remain_cap, 0)\n    )\n    \n    # Add noise to break ties and improve exploration slightly\n    np.random.seed(hash(str(item)) % 100 + int(time.time() * 1000) % 100)  # Syntactic: hash-based seed to impact randomness\n    noise = np.random.normal(0, 0.2, score.shape)\n    \n    return score + noise\n"


def get_seed_code() -> str:
    """返回内联的种子源码字符串。"""
    return SEED_CODE.strip()

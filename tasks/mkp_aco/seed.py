"""
内联种子代码。
"""

SEED_CODE = 'import numpy as np\n\ndef heuristics_v1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:\n    return prize / np.sum(weight, axis=1)'


def get_seed_code() -> str:
    """返回内联的种子源码字符串。"""
    return SEED_CODE.strip()

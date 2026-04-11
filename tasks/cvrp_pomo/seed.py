"""
内联种子代码。
"""

SEED_CODE = 'import torch\n\ndef heuristics_v1(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:\n    """A trivial implementation to improve upon."""\n    return torch.zeros_like(distance_matrix)'


def get_seed_code() -> str:
    """返回内联的种子源码字符串。"""
    return SEED_CODE.strip()

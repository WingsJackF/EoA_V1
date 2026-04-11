"""
内联种子代码。
"""

SEED_CODE = 'import torch\ndef heuristics_v2(distance_matrix: torch.Tensor) -> torch.Tensor:\n    """\n    heu_ij = - log(dis_ij) if j is the topK nearest neighbor of i, else - dis_ij\n    """\n    distance_matrix[distance_matrix == 0] = 1e5\n    K = 100\n    # Compute top-k nearest neighbors (smallest distances)\n    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)\n    heu = -distance_matrix.clone()\n    # Create a mask where topk indices are True and others are False\n    topk_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)\n    topk_mask.scatter_(1, indices, True)\n    # Apply -log(d_ij) only to the top-k elements\n    heu[topk_mask] = -torch.log(distance_matrix[topk_mask])\n    return heu\n'


def get_seed_code() -> str:
    """返回内联的种子源码字符串。"""
    return SEED_CODE.strip()

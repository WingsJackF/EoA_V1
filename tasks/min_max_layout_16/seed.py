"""任务专用：种子代码与策略说明（与具体几何目标绑定）。"""

SEED_CODE = """
import numpy as np

def min_max_dist_dim2_16() -> np.ndarray:
    '''
    Construct a 16-point 2D layout based on a hexagonal (triangular) lattice.
    Steps:
    - Create a small hexagonal lattice patch (5x5 indices).
    - Select the 16 lattice sites nearest the lattice center (most central).
    - Recenter the selected points at the origin.
    - Scale so the minimum pairwise distance equals 1 (for numerical convenience).
    Returns:
        points: np.ndarray of shape (16, 2)
    '''
    sqrt3 = np.sqrt(3.0)
    # Build a 5x5 hexagonal/triangular lattice patch
    coords = []
    grid_size = 5
    for j in range(grid_size):
        for i in range(grid_size):
            # x offset by 0.5 per row gives the triangular/hex packing
            x = i + 0.5 * j
            y = j * (sqrt3 / 2.0)
            coords.append((x, y))
    coords = np.array(coords, dtype=float)

    # Choose the 16 points closest to the patch center (most central 16)
    center = coords.mean(axis=0)
    distances_to_center = np.linalg.norm(coords - center, axis=1)
    chosen_idx = np.argsort(distances_to_center)[:16]
    points = coords[chosen_idx]

    # Recenter at origin
    points = points - points.mean(axis=0)

    # Compute pairwise distances and scale so minimum pairwise distance == 1.0
    diffs = points[:, None, :] - points[None, :, :]
    pairwise = np.linalg.norm(diffs, axis=2)
    # Avoid the zero diagonal interfering
    np.fill_diagonal(pairwise, np.inf)
    min_dist = pairwise.min()
    if min_dist > 0:
        points = points / min_dist

    return points.astype(float)
"""

SEED_THOUGHT = (
    "Use a compact hexagonal (triangular) lattice — the densest planar packing — and carve out "
    "the 16 most central lattice sites. The hexagonal lattice maximizes nearest-neighbor spacing "
    "for a given overall diameter, so selecting the 16 points closest to the lattice center "
    "produces a compact, well-separated cluster that tends to increase the minimum-to-maximum "
    "pairwise distance ratio. To make the layout scale-insensitive and numerically convenient, "
    "recenter the selected points at the origin and scale them so the smallest pairwise distance "
    "equals 1. This construction is deterministic and reproducible."
)

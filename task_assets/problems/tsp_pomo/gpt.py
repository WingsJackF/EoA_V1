import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    Enhanced TSP heuristics using pure rank-based measures, MST-based filtering,
    and triangle inequality analysis with adaptive percentile thresholds.
    
    Key improvements:
    1. Pure rank-based measures without distance normalization artifacts
    2. Enhanced clustering analysis with triangle inequality consideration
    3. Adaptive MST thresholds based on distribution properties
    4. Better handling of symmetry and mutual preferences
    
    Returns heuristic tensor with zero mean where positive values indicate promising edges.
    """
    n = distance_matrix.shape[0]
    device = distance_matrix.device
    eps = 1e-12
    
    # Create working copy of distance matrix
    dist = distance_matrix.clone()
    
    # Replace zeros (diagonal) with large value to avoid division issues
    eye_mask = torch.eye(n, device=device, dtype=torch.bool)
    dist[eye_mask] = dist[dist > 0].max() if dist[dist > 0].numel() > 0 else 1.0
    dist[dist == 0] = eps
    
    # 1. Pure rank-based mutual preference scores
    # Get ranks for each node's outgoing edges (lower rank = smaller distance)
    row_ranks = torch.argsort(torch.argsort(dist, dim=1), dim=1).float()
    # Convert to preference scores (1 for nearest neighbor, 0 for farthest)
    row_pref = 1.0 - (row_ranks / (n - 1))
    
    # Get ranks for incoming edges (symmetry)
    col_ranks = torch.argsort(torch.argsort(dist, dim=0), dim=0).float()
    col_pref = 1.0 - (col_ranks / (n - 1))
    
    # Mutual rank preference - geometric mean for stronger symmetry requirement
    mutual_rank = torch.sqrt(row_pref * col_pref + eps)
    
    # 2. MST-inspired edge filtering with adaptive percentiles
    # Use only non-diagonal distances
    mask = ~eye_mask
    flat_dist = dist[mask]
    
    mst_score = torch.zeros_like(dist)
    
    if len(flat_dist) > 0:
        # Calculate adaptive percentiles based on distance distribution
        q1 = torch.quantile(flat_dist, 0.25)
        median = torch.quantile(flat_dist, 0.5)
        q3 = torch.quantile(flat_dist, 0.75)
        
        iqr = q3 - q1
        # Adaptive thresholds based on distribution spread
        low_thresh = torch.clamp(q1 - 0.5 * iqr, flat_dist.min(), flat_dist.max())
        high_thresh = torch.clamp(q3 + 1.5 * iqr, flat_dist.min(), flat_dist.max())
        
        # Assign scores based on percentile ranges
        # High priority for edges likely to be in MST
        mst_score[(dist <= low_thresh) & mask] = 1.0
        
        # Medium priority for edges in lower-mid range
        mid_low_mask = (dist > low_thresh) & (dist <= median) & mask
        mst_score[mid_low_mask] = 0.7
        
        # Lower priority for edges in upper-mid range
        mid_high_mask = (dist > median) & (dist <= high_thresh) & mask
        mst_score[mid_high_mask] = 0.3
        
        # Penalize outlier edges (unlikely to be in good tours)
        outlier_mask = (dist > high_thresh) & mask
        mst_score[outlier_mask] = -0.5
    
    # 3. Triangle inequality satisfaction analysis
    tri_score = torch.zeros_like(dist)
    
    # For each edge, check if it satisfies triangle inequality with other nodes
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle, we'll symmetrize later
            if i != j:
                # Find other nodes k
                k_mask = torch.ones(n, dtype=torch.bool, device=device)
                k_mask[i] = False
                k_mask[j] = False
                
                if k_mask.sum() > 0:
                    # Check triangle inequality: dist[i,j] <= dist[i,k] + dist[k,j]
                    ik_dist = dist[i, k_mask]
                    kj_dist = dist[k_mask, j]
                    triangle_sums = ik_dist + kj_dist
                    
                    # Count how many triangles are satisfied
                    satisfied = (dist[i, j] <= triangle_sums).float()
                    tri_score[i, j] = satisfied.mean()
    
    # Make triangle scores symmetric
    tri_score = tri_score + tri_score.T
    tri_score[eye_mask] = 0
    
    # 4. Global distance percentile rank
    rank_map = torch.zeros_like(dist)
    if len(flat_dist) > 0:
        # Sort all distances for global percentile calculation
        sorted_dists, _ = torch.sort(flat_dist)
        
        # Calculate percentile for each non-diagonal distance
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Find percentile position in sorted distances
                    idx = torch.searchsorted(sorted_dists, dist[i, j])
                    percentile = idx.float() / len(sorted_dists)
                    # Convert to preference (higher for smaller distances)
                    rank_map[i, j] = 1.0 - percentile
    
    # 5. Combine components with optimized weights
    # Base combination: mutual ranks + MST scores + triangle scores + global ranks
    heu = (0.40 * mutual_rank +      # Mutual nearest neighbor ranks
           0.30 * mst_score +        # MST likelihood
           0.20 * tri_score +        # Triangle inequality satisfaction
           0.10 * rank_map)          # Global percentile
    
    # 6. Boost edges that excel in multiple criteria
    strong_mask = (mutual_rank > 0.8) & (mst_score > 0.6) & (tri_score > 0.7)
    heu[strong_mask] *= 1.5
    
    # Penalize edges that are weak across criteria
    weak_mask = (mutual_rank < 0.2) & (mst_score < 0.1) & (tri_score < 0.3)
    heu[weak_mask] *= 0.3
    
    # 7. Ensure symmetry for undirected TSP
    heu = (heu + heu.T) / 2.0
    
    # 8. Robust normalization to zero mean
    # Remove diagonal from normalization
    heu_vals = heu[mask]
    
    if len(heu_vals) > 0:
        # Calculate robust statistics
        heu_median = torch.median(heu_vals)
        heu_mad = torch.median(torch.abs(heu_vals - heu_median)) + eps
        
        # Normalize using robust scale estimate
        heu_normalized = (heu - heu_median) / heu_mad
        
        # Recenter to have zero mean
        heu_normalized_mean = heu_normalized[mask].mean()
        heu = heu_normalized - heu_normalized_mean
    else:
        # Fallback if no valid edges
        heu = torch.zeros_like(dist)
    
    # 9. Strong penalty for self-loops (diagonal)
    heu[eye_mask] = -1e9
    
    return heu

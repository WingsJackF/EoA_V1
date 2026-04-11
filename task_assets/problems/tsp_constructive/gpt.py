import numpy as np
def select_next_node_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """Select the next node using balanced, interpretable heuristics that prioritize local cost, diversity, and destination convergence."""
    if len(unvisited_nodes) == 1:
        return unvisited_nodes.pop()
    
    # Smooth, interpretable weights that favor local cost and diversity, while guiding toward destination
    c_local = 0.6     # Strong preference for nearby nodes to reduce immediate cost
    c_diversity = 0.3 # Encourage visiting nodes that explore new regions
    c_convergence = 0.1 # Mild preference for nodes closer to destination for timely return
    
    scores = {}
    
    for node in unvisited_nodes:
        # 1. Local cost: direct edge cost from current node
        local_cost = distance_matrix[current_node][node]
        
        # 2. Diversity: encourage nodes that are neither too close nor too far from others (spread across unvisited nodes)
        others = [n for n in unvisited_nodes if n != node]
        if len(others) > 1:
            distances_to_others = [distance_matrix[node][n] for n in others]
            std_dist = np.std(distances_to_others)
            diversity_score = std_dist
        else:
            diversity_score = 0
        
        # 3. Convergence: estimate of return cost from this node to destination
        dist_to_dest = distance_matrix[node][destination_node]
        
        # Composite score: lower is better
        # Lower local cost is better, higher diversity helps exploration, lower return cost is better
        score = (
            c_local * local_cost -
            c_diversity * diversity_score +
            c_convergence * dist_to_dest
        )
        
        scores[node] = score
    
    # Select the node with the minimum score
    return min(scores, key=scores.get)

import numpy as np
import numpy as np

def crossover_v2(parents: np.ndarray, n_pop: int) -> np.ndarray:
    """
    Simplified yet effective crossover function for decap placement optimization.
    Features:
    1. Multi-point crossover with random crossover points
    2. Uniform crossover with fixed probability
    3. Vectorized implementation for efficiency
    4. Random parent selection without tournament overhead
    """
    n_parents, n_decap = parents.shape
    
    # Parameters
    uniform_crossover_prob = 0.3
    blend_alpha = 0.4
    
    # Initialize offspring array
    offspring = np.zeros((n_pop, n_decap), dtype=parents.dtype)
    
    # Generate all random selections upfront for vectorization
    parent_indices = np.random.randint(0, n_parents, size=(n_pop, 2))
    parent1_indices, parent2_indices = parent_indices[:, 0], parent_indices[:, 1]
    
    # Get parent pairs
    parent1 = parents[parent1_indices]
    parent2 = parents[parent2_indices]
    
    # Pre-generate random strategy choices for each offspring
    strategies = np.random.rand(n_pop)
    
    # Pre-generate uniform crossover masks for all offspring
    uniform_masks = np.random.rand(n_pop, n_decap) < uniform_crossover_prob
    
    for i in range(n_pop):
        p1, p2 = parent1[i], parent2[i]
        strategy = strategies[i]
        
        if strategy < 0.5:
            # Multi-point crossover (1-3 points)
            n_points = np.random.randint(1, min(4, n_decap))
            points = np.sort(np.random.choice(range(1, n_decap), n_points, replace=False))
            points = np.concatenate([[0], points, [n_decap]])
            
            child = np.empty(n_decap, dtype=parents.dtype)
            use_p1 = True
            
            for start, end in zip(points[:-1], points[1:]):
                if use_p1:
                    child[start:end] = p1[start:end]
                else:
                    child[start:end] = p2[start:end]
                use_p1 = not use_p1
                
        elif strategy < 0.8:
            # Uniform crossover
            mask = uniform_masks[i]
            child = np.where(mask, p1, p2)
            
        else:
            # Blend crossover
            if np.all(np.isin(parents, [0, 1])):  # Binary case
                blend = blend_alpha * p1 + (1 - blend_alpha) * p2
                child = (blend > 0.5).astype(parents.dtype)
            else:  # Continuous case
                child = blend_alpha * p1 + (1 - blend_alpha) * p2
        
        offspring[i] = child
    
    return offspring

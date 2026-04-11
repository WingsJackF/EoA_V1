import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: implement_elitism_selection

Description:
    Implements the elitism selection strategy by selecting the top N individuals
    from a population according to their fitness['combined_score']. Returns deep
    copies of the selected elites to prevent downstream modification.

Constraints:
    - Python 3.9+
    - No file I/O.
    - No modification of individuals in-place.
    - Uses specific exception types only when validating inputs.
"""

from typing import List, Dict, Any
import copy


def implement_elitism_selection(
    population: List[Dict[str, Any]],
    n_elites: int
) -> List[Dict[str, Any]]:
    '''
    Selects the top n_elites individuals by combined_score for elitism.

    Args:
        population: List of individual dictionaries (each must have 'fitness' with 'combined_score').
        n_elites: Number of elite individuals to select (non-negative integer).

    Returns:
        A list of elite individuals (deep copies).

    Raises:
        TypeError: If population is not a list or n_elites is not an integer.
        ValueError: If n_elites is negative.
        KeyError: If an individual's fitness structure is missing required keys.
    '''
    # Input validation
    if not isinstance(population, list):
        raise TypeError("population must be a list of individual dictionaries")
    if not isinstance(n_elites, int):
        raise TypeError("n_elites must be an integer")
    if n_elites < 0:
        raise ValueError("n_elites must be non-negative")

    # If population is empty or n_elites is zero, return empty list
    if len(population) == 0 or n_elites == 0:
        return []

    # Define a helper to safely extract combined_score; missing values default to 0.0
    def _combined_score_of(ind: Dict[str, Any]) -> float:
        fitness = ind.get("fitness", {})
        # If fitness is not a dict, this will cause a TypeError naturally.
        try:
            cs = fitness.get("combined_score", 0.0)
        except AttributeError:
            # fitness isn't a dict-like object
            raise KeyError("Individual fitness must be a dict containing 'combined_score'")
        # Convert to float if possible
        try:
            return float(cs)
        except (TypeError, ValueError):
            # If conversion fails, treat as 0.0
            return 0.0

    # Sort population by combined_score descending
    sorted_pop = sorted(population, key=_combined_score_of, reverse=True)

    # Determine how many elites to return
    n_select = n_elites if n_elites < len(sorted_pop) else len(sorted_pop)

    elites = [copy.deepcopy(sorted_pop[i]) for i in range(n_select)]
    return elites



import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: implement_tournament_selection

Description:
    Implements tournament selection for evolutionary reproduction by running
    contests among random subsets of the current population. Winners (highest
    fitness by 'combined_score') are returned as deep copies.

Constraints and Notes:
    - Python 3.9+
    - No file I/O.
    - No broad exception swallowing; only specific exceptions handled.
    - Does not modify individuals in-place (returns deep copies).
"""

from typing import List, Dict, Any
import random
import copy


def implement_tournament_selection(
    population: List[Dict[str, Any]],
    n_selections: int,
    tournament_size: int = 3
) -> List[Dict[str, Any]]:
    '''
    Selects individuals via tournament selection.

    Args:
        population: List of individual dictionaries (must each have 'fitness' with 'combined_score').
        n_selections: Number of individuals to select (non-negative integer).
        tournament_size: Number of contestants per tournament (positive integer).

    Returns:
        A list of selected individuals (deep copies).

    Raises:
        TypeError: If population is not a list or n_selections/tournament_size are not integers.
        ValueError: If n_selections is negative or tournament_size is less than 1.
        KeyError: If an individual's fitness structure is missing required keys.
    '''
    # Input validation
    if not isinstance(population, list):
        raise TypeError("population must be a list of individual dictionaries")
    if not isinstance(n_selections, int):
        raise TypeError("n_selections must be an integer")
    if not isinstance(tournament_size, int):
        raise TypeError("tournament_size must be an integer")
    if n_selections < 0:
        raise ValueError("n_selections must be non-negative")
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1")

    # Early returns
    if n_selections == 0 or len(population) == 0:
        return []

    def _combined_score_of(ind: Dict[str, Any]) -> float:
        """
        Safely extract combined_score from an individual's fitness dictionary.
        Missing or non-convertible values are treated as 0.0. If fitness is not dict-like,
        a KeyError is raised to indicate malformed individual.
        """
        fitness = ind.get("fitness", {})
        if not isinstance(fitness, dict):
            raise KeyError("Individual fitness must be a dict containing 'combined_score'")
        cs = fitness.get("combined_score", 0.0)
        try:
            return float(cs)
        except (TypeError, ValueError):
            return 0.0

    selected: List[Dict[str, Any]] = []

    # For each selection, run a tournament among randomly sampled contestants.
    for _ in range(n_selections):
        k = tournament_size if tournament_size <= len(population) else len(population)
        # random.sample requires k <= len(population); using k computed above ensures that.
        contestants = random.sample(population, k=k)
        # Determine winner by highest combined_score
        winner = max(contestants, key=_combined_score_of)
        selected.append(copy.deepcopy(winner))

    return selected




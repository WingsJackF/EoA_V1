import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: archive_best_individuals_module

Description:
    Implements an archival module to track and store historically best-performing
    individuals. Combines incoming candidates with existing archive, deduplicates
    by (thought, code) tuple, sorts by fitness['combined_score'] descending, trims
    to max_archive_size, and returns deep copies to ensure immutability.

Constraints:
    - Python 3.9+
    - No file I/O.
    - Uses typing and copy from the standard library.
    - Does not perform in-place mutation of archived individuals.
"""

from typing import List, Dict, Any, Tuple
import copy

from tasks.base import fitness_has_error


def _extract_combined_score(ind: Dict[str, Any]) -> float:
    """
    Safely extract combined_score from an individual's fitness dict.
    If missing or invalid, returns 0.0.
    Raises KeyError if fitness is not a dict.
    """
    fitness = ind.get("fitness", {})
    if not isinstance(fitness, dict):
        raise KeyError("Individual fitness must be a dict containing 'combined_score'")
    cs = fitness.get("combined_score", 0.0)
    try:
        return float(cs)
    except (TypeError, ValueError):
        return 0.0


def _archive_sort_key(ind: Dict[str, Any]) -> Tuple[float, int]:
    """主键 combined_score 降序；同分优先无 error。"""
    try:
        score = _extract_combined_score(ind)
    except KeyError:
        score = 0.0
    has_error = 1 if fitness_has_error(ind.get("fitness", {})) else 0
    return (score, -has_error)


def archive_best_individuals(
    archive: List[Dict[str, Any]],
    new_candidates: List[Dict[str, Any]],
    max_archive_size: int = 10
) -> List[Dict[str, Any]]:
    '''
    Updates the archive with new best-performing individuals.

    Args:
        archive: Current archive (list of individuals).
        new_candidates: List of new individuals to consider for archival.
        max_archive_size: Maximum number of individuals to keep in the archive.

    Returns:
        Updated archive list (deep copies).

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If max_archive_size is not a positive integer.
    '''
    if not isinstance(archive, list):
        raise TypeError("archive must be a list of individual dictionaries")
    if not isinstance(new_candidates, list):
        raise TypeError("new_candidates must be a list of individual dictionaries")
    if not isinstance(max_archive_size, int):
        raise TypeError("max_archive_size must be an integer")
    if max_archive_size <= 0:
        raise ValueError("max_archive_size must be a positive integer")

    combined = [
        ind
        for ind in (list(archive) + list(new_candidates))
        if isinstance(ind, dict) and not fitness_has_error(ind.get("fitness", {}))
    ]

    # Deduplicate by (thought, code). Keep the version with the highest combined_score.
    unique_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for ind in combined:
        # Ensure structure is a dict
        if not isinstance(ind, dict):
            # Skip malformed entries rather than raising to be robust
            continue

        thought = ind.get("thought", "")
        code = ind.get("code", "")
        # Normalize keys to strings to make tuple hashing robust
        key = (str(thought), str(code))

        # Determine combined_score safely
        try:
            cs = _extract_combined_score(ind)
        except KeyError:
            # Treat missing fitness structure as score 0.0
            cs = 0.0

        existing = unique_map.get(key)
        if existing is None:
            unique_map[key] = ind
        else:
            try:
                existing_cs = _extract_combined_score(existing)
            except KeyError:
                existing_cs = 0.0
            if cs > existing_cs:
                unique_map[key] = ind
            elif cs == existing_cs and fitness_has_error(existing.get("fitness", {})) and not fitness_has_error(ind.get("fitness", {})):
                unique_map[key] = ind

    unique_list = list(unique_map.values())
    sorted_list = sorted(unique_list, key=_archive_sort_key, reverse=True)

    # Trim to max_archive_size
    trimmed = sorted_list[:max_archive_size]

    # Return deep copies to prevent downstream modification
    return [copy.deepcopy(ind) for ind in trimmed]



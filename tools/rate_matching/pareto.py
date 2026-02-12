"""
Pareto frontier extraction for rate-matching results.

Extracts the 2D Pareto-optimal set from rate-matching results where:
  - X axis = interactivity (tok/s/user) -- higher is better
  - Y axis = output_tput_per_gpu (tok/s/GPU) -- higher is better

Points are ranked by interactivity descending (rank 1 = most interactive).
"""

from __future__ import annotations


def extract_pareto_frontier(results: list[dict]) -> list[dict]:
    """Extract Pareto-optimal configurations from rate-matching results.

    A point is Pareto-optimal if no other point has both higher interactivity
    AND higher output_tput_per_gpu.

    Args:
        results: List of dicts from compute_rate_matching(), each containing
                 at least 'interactivity' and 'output_tput_per_gpu'.

    Returns:
        Sorted list of Pareto-optimal points with 'pareto_rank' and
        'is_pareto_optimal' fields added. Ranked by interactivity descending.
    """
    if not results:
        return []

    # Filter out any results with errors
    valid = [r for r in results if 'error' not in r and r.get('interactivity', 0) > 0]
    if not valid:
        return []

    # Sort by interactivity descending (primary), tput_per_gpu descending (secondary)
    sorted_results = sorted(
        valid,
        key=lambda x: (-x.get('interactivity', 0), -x.get('output_tput_per_gpu', 0)),
    )

    # Extract Pareto frontier
    # Walk from highest interactivity to lowest.
    # A point is Pareto-optimal if its tput_per_gpu is higher than all
    # previously seen points (which all have higher interactivity).
    frontier = []
    max_tput_seen = -1.0

    for r in sorted_results:
        tput = r.get('output_tput_per_gpu', 0)
        if tput > max_tput_seen:
            frontier.append(r)
            max_tput_seen = tput

    # Assign ranks (1 = highest interactivity)
    for rank, point in enumerate(frontier, 1):
        point['pareto_rank'] = rank
        point['is_pareto_optimal'] = True

    return frontier

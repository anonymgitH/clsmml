from typing import List
def calculate_accuracy(true_directions: List[str], estimated_directions: List[str], weights: List[float]) -> float:
    """
    Calculate the accuracy of causal direction decisions.
    """
    assert len(true_directions) == len(estimated_directions) == len(weights), "All input lists must have the same length"

    numerator = 0
    denominator = 0

    for true_dir, est_dir, weight in zip(true_directions, estimated_directions, weights):
        if est_dir != '?':  # Only consider decisions that are not undecided
            if true_dir == est_dir:
                numerator += weight
            denominator += weight

    if denominator == 0:
        return 0.0  

    accuracy = numerator / denominator
    return accuracy

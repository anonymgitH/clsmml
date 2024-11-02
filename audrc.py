
import numpy as np

def calculate_AUDRC(ground_truth, estimator, certainty):
    """
    Calculate the Area Under the Decision-Related ROC Curve (AUDRC).

    """
    M = len(ground_truth)
    if M != len(estimator) or M != len(certainty):
  
        raise ValueError("Lengths of ground_truth, estimator, and certainty must be the same.")

    sorted_indices = sorted(range(M), key=lambda i: certainty[i], reverse=True)
   
    AUDRC = sum(1 for i in range(M) if estimator[sorted_indices[i]] == ground_truth[sorted_indices[i]]) / M

    return AUDRC

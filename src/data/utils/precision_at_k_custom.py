#src/data/precision_at_k.py

#Calculates the precision score of top k hits for a given model output

import torch
import numpy as np

def precision_at_k_custom(proba_matrix, targets, k=5):
    """
    Takes the prediction matrix of probabilities and calculates the precision@k.

    Args:
        proba_matrix (torch.tensor) : probability matrix of prediction of size [n_test_samples,n_labels]
        targets (torch.tensor) : Actual labels y_test (transformed/binarized) of size [n_test_samples,n_labels]
        k (int) : top k predictions

    Returns:
        precision@k score (float) : Averages the precision@k score over all test samples 

    """
    _, topk_indices = torch.topk(proba_matrix,k=k,dim=1)
    topk_targets = torch.gather(targets, dim=1, index=topk_indices)

    # Average precision across all samples
    precision_per_sample = topk_targets.sum(dim=1) / float(k)

    return precision_per_sample.mean().item()
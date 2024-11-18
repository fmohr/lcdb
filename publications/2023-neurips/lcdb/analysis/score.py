import warnings

import numpy as np


def accuracy_from_confusion_matrix(x: np.ndarray) -> float:
    """Compute the accuracy from a confusion matrix.

    Args:
        x (np.ndarray): the confusion matrix.

    Returns:
        float: the balanced accuracy.
    """
    return np.diag(x).sum() / np.sum(x)


def balanced_accuracy_from_confusion_matrix(x: np.ndarray) -> float:
    """Compute the balanced accuracy from a confusion matrix.

    Code from: https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/metrics/_classification.py#L2326

    Args:
        x (np.ndarray): the confusion matrix.

    Returns:
        float: the balanced accuracy.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(x) / np.sum(x, axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn("y_pred contains classes not in y_true")
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


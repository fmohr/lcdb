import numpy as np


def accuracy_from_confusion_matrix(x):
    """Compute the accuracy from a confusion matrix."""
    return np.diag(x).sum() / np.sum(x)

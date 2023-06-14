import sys
from typing import Tuple

import numpy as np


def _load_iris() -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load iris dataset

    Returns:
        (np.ndarray, np.ndarray, dict): X, y, metadata where X is the input array, y is the target array, and metadata is a dictionary containing additional information about the data.
    """
    from sklearn.datasets import load_iris

    iris = load_iris()

    X = iris["data"]
    y = iris["target"]

    metadata = {
        "type": "classification",
        "input_dimension": X.shape[1],
        "num_classes": len(iris["target_names"]),
        "description": {k: v for k, v in iris.items() if k not in ["data", "target"]}
    }
    
    return (X, y), metadata


def load_from_sklearn(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load dataset from sklearn library.

    Args:
        dataset_name (str): the name of the dataset to be loaded.

    Raises:
        ValueError: if the dataset is not found.

    Returns:
        (np.ndarray, np.ndarray, dict): X, y, metadata where X is the input array, y is the target array, and metadata is a dictionary containing additional information about the data.
    """
    try:
        load_function = getattr(sys.modules[__name__], f"_load_{dataset_name}")
    except AttributeError:
        raise ValueError(f"Dataset {dataset_name} not found.")

    data, metadata = load_function()

    # Verifications of metadata format
    assert "type" in metadata
    if metadata["type"] == "classification":
        assert "input_dimension" in metadata
        assert "num_classes" in metadata

    return data, metadata

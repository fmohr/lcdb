from typing import Tuple

import numpy as np
import openml


def load_from_openml(dataset_id: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load dataset from openml library.

    Args:
        dataset_id (str): the identifier of the dataset to be loaded from the OpenML database.

    Raises:
        ValueError: if the dataset is not found.

    Returns:
        (np.ndarray, np.ndarray, dict): X, y, metadata where X is the input array, y is the target array, and metadata is a dictionary containing additional information about the data.
    """
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )

    X, y, categorical_indicator, _ = dataset.get_data(
        target=dataset.default_target_attribute
    )

    X, y = X.values, y.values

    metadata = {
        "type": "classification",
        "num_classes": len(dataset.retrieve_class_labels()),
        "categories": categorical_indicator,
        "description": dataset.description,
    }

    return (X, y), metadata


if __name__ == "__main__":
    # Some tests
    (X, y), metadata = load_from_openml("3")
    print(metadata["description"])

from sklearn.preprocessing import LabelEncoder


def load_task(task_name: str):
    """Loads a task by name.

    Args:
        task_name (str): Name of the task to load. A task name is composed of the `source` and the `name` of the task, separated by a dot. For example, `sklearn.breast_cancer` loads the `breast_cancer` task from `sklearn`. Similarly, `openml.3` loads the `3` task from `openml`.

    Returns:
        data (tuple): Tuple of `(X, y)` arrays.
        metadata (dict): Dictionary of metadata for the task.
    """
    if task_name.startswith("sklearn"):
        from lcdb.data._sklearn import load_from_sklearn

        data, metadata = load_from_sklearn(task_name[len("sklearn.") :])
        metadata["name"] = task_name
    elif task_name.startswith("openml"):
        from lcdb.data._openml import load_from_openml

        data, metadata = load_from_openml(task_name[len("openml.") :])
        metadata["name"] = task_name
    else:
        raise ValueError(f"Unknown task '{task_name}'")

    # Verifications of metadata format
    assert "type" in metadata
    if metadata["type"] == "classification":
        assert "num_classes" in metadata

        X, y = data
        y = LabelEncoder().fit_transform(y)
        data = (X, y)

    return data, metadata


if __name__ == "__main__":
    # Some tests
    import numpy as np

    (X, y), metadata = load_task("sklearn.iris")
    print(np.shape(X), np.shape(y))
    print(metadata)

    (X, y), metadata = load_task("openml.3")
    print(np.shape(X), np.shape(y))
    print(metadata)

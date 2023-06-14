def load_task(task_name: str):
    if task_name.startswith("sklearn"):
        from ._sklearn import load_from_sklearn

        data, metadata = load_from_sklearn(task_name[len("sklearn.") :])
        metadata["name"] = task_name
    else:
        raise ValueError(f"Unknown task '{task_name}'")

    return data, metadata

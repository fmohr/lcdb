import importlib


def import_attr_from_module(path: str):
    """Import an attribute from a module given its path.

    Example:

    >>> from lcdb.workflow import SVMWorkflow
    >>> import_attr_from_module("lcdb.workflow.SVMWorkflow") == SVMWorkflow
    """
    path = path.split(".")
    module_name, attr_name = (
        ".".join(path[:-1]),
        path[-1],
    )
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    return attr


def filter_keys_with_prefix(d: dict, prefix: str) -> dict:
    """Filter keys from a dictionary that start with a given prefix.
    Then the fildered dictionnary with prefix removed from the keys is returned.

    Example:

    >>> filter_keys_with_prefix({"p:a": 1, "p:b": 2, "c": 3}, prefix="p:")
    {"a": 1, "b": 2}

    Args:
        d (dict): the dictionary to filter.
        prefix (str): the prefix to use for filtering.

    Returns:
        dict: the filtered dictionary.
    """
    return {k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)}

import importlib
import multiprocessing
import multiprocessing.pool
import time
import numpy as np


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


class FunctionCallTimeoutError(Exception):
    """Exception raised when a function call times out."""


def terminate_on_timeout(timeout, func, *args, **kwargs):
    """High order function to wrap the call of a function in a thread to monitor its execution time.

    >>> import functools
    >>> f_timeout = functools.partial(terminate_on_timeout, 10, f)
    >>> f_timeout(1, b=2)

    Args:
        timeout (int): timeout in seconds.
        func (function): function to call.
        *args: positional arguments to pass to the function.
        **kwargs: keyword arguments to pass to the function.
    """

    pool = multiprocessing.pool.ThreadPool(processes=1)
    results = pool.apply_async(func, args, kwargs)
    pool.close()
    try:
        return results.get(timeout)
    except multiprocessing.TimeoutError:
        raise FunctionCallTimeoutError(f"Function timeout expired after: {timeout}")
    finally:
        pool.terminate()


def get_schedule(name, **kwargs):
    """Get a schedule given its name and optional arguments.

    Args:
        name (str): name of the schedule.
        **kwargs: optional arguments to pass to the schedule.
    """
    if name == "linear":
        return get_linear_schedule(**kwargs)
    elif name == "last":
        return [kwargs["n"]]
    elif name == "power":
        return get_power_schedule(**kwargs)
    else:
        raise ValueError(f"Unknown schedule: {name}")


def get_linear_schedule(n: int, step: 1 = 1, **kwargs):
    return list(range(1, n + 1, step))


def get_power_schedule(n: int, base=2, power=0.5, delay: int = 7, **kwargs):
    """Get a schedule of anchors for a given size `n`."""
    anchors = []
    k = 1
    while True:
        exponent = (delay + k) * power
        sample_size = int(np.round(base**exponent))
        if sample_size > n:
            break
        anchors.append(sample_size)
        k += 1
    if anchors[-1] < n:
        anchors.append(n)
    return anchors

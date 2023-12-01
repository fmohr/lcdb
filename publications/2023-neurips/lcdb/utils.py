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


def get_anchor_schedule(n: int, delay: int = 7):
    """Get a schedule of anchors for a given size `n`."""
    anchors = []
    k = 1
    while True:
        exponent = (delay + k) / 2
        sample_size = int(np.round(2**exponent))
        if sample_size > n:
            break
        anchors.append(sample_size)
        k += 1
    if anchors[-1] < n:
        anchors.append(n)
    return anchors


def get_iteration_schedule(n: int) -> list:
    """Get a schedule of iterations for a given size `n`."""
    return get_anchor_schedule(n, delay=0)

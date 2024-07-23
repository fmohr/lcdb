import functools
import importlib
import multiprocessing
import multiprocessing.pool
import os
import signal
import time
from concurrent.futures import CancelledError, ProcessPoolExecutor, BrokenExecutor

import numpy as np
from scipy.special import softmax
import psutil

from deephyper.evaluator._run_function_utils import standardize_run_function_output


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


def terminate_on_memory_exceeded(
    memory_limit,
    memory_tracing_interval,
    raise_exception,
    func,
    *args,
    **kwargs,
):
    """Decorator to use on a ``run_function`` to profile its execution-time and peak memory usage.

    Args:
        memory_limit (int): In bytes, if set to a positive integer, the memory usage is measured at regular intervals and the function is interrupted if the memory usage exceeds the limit. If set to ``-1``, only the peak memory is measured. If the executed function is busy outside of the Python interpretor, this mechanism will not work properly. Defaults to ``-1``.
        memory_tracing_interval (float): In seconds, the interval at which the memory usage is measured. Defaults to ``0.1``.

    Returns:
        function: a decorated function.
    """

    timestamp_start = time.time()

    p = psutil.Process()  # get the current process

    output = None

    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(os.getpid)
            pid = future.result()
            p = psutil.Process(pid)

            future = executor.submit(func, *args, **kwargs)

            memory_peak = p.memory_info().rss

            while not future.done():

                # in bytes (not the peak memory but last snapshot)
                memory_peak = max(p.memory_info().rss, memory_peak)

                if memory_limit > 0 and memory_peak > memory_limit:
                    output = "F_memory_limit_exceeded"
                    os.kill(pid, signal.SIGTERM)
                    future.cancel()

                    if raise_exception:
                        raise CancelledError(
                            f"Memory limit exceeded: {memory_peak} > {memory_limit}"
                        )

                    break

                time.sleep(memory_tracing_interval)

            if output is None:
                output = future.result()
    except BrokenExecutor:
        pass

    timestamp_end = time.time()

    output = standardize_run_function_output(output)
    metadata = {
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }

    metadata["memory"] = memory_peak

    metadata.update(output["metadata"])
    output["metadata"] = metadata

    return output


def get_schedule(name, **kwargs):
    """Get a schedule given its name and optional arguments.

    Args:
        name (str): name of the schedule.
        **kwargs: optional arguments to pass to the schedule.
    """
    if name == "full":
        return get_linear_schedule(**kwargs)
    elif name == "linear":
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


def decision_fun_to_proba(decision_fun_vals):
    """
    take a vector or matrix of decision function values and turn them into probabilities through a softmax

    :param decision_fun_vals:
    :return:
    """
    sigmoid = lambda z: 1/(1 + np.exp(-z))
    if len(decision_fun_vals.shape) == 2:
        return softmax(decision_fun_vals, axis=1)
    else:  # if the decision function values is only a vector, then these are the probs of the positive class
        a = sigmoid(decision_fun_vals)
        return np.column_stack([1 - a, a])


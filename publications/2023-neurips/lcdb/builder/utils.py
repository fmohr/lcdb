import jsonlines
import pathlib
import logging
import multiprocessing
import multiprocessing.pool
import os
import signal
import time
from concurrent.futures import BrokenExecutor, CancelledError, ProcessPoolExecutor
from numbers import Number
from typing import Union

import pandas as pd
import numpy as np
import psutil
from scipy.special import softmax

import traceback

def standardize_run_function_output(
    output: Union[str, float, tuple, list, dict],
) -> dict:
    """Transform the output of the run-function to its standard form.

    Possible return values of the run-function are:

    >>> 0
    >>> 0, 0
    >>> "F_something"
    >>> {"objective": 0 }
    >>> {"objective": (0, 0), "metadata": {...}}

    Args:
        output (Union[str, float, tuple, list, dict]): the output of the run-function.

    Returns:
        dict: standardized output of the function.
    """

    # output returned a single objective value
    if np.isscalar(output):
        if isinstance(output, str):
            output = {"objective": output}
        elif isinstance(output, Number):
            output = {"objective": float(output)}
        else:
            raise TypeError(
                f"The output of the run-function cannot be of type {type(output)} it should be either a string or a number."
            )

    # output only returned objective values as tuple or list
    elif isinstance(output, (tuple, list)):

        output = {"objective": output}

    elif isinstance(output, dict):
        pass
    else:
        raise TypeError(
            f"The output of the run-function cannot be of type {type(output)}"
        )

    metadata = output.get("metadata", dict())
    if metadata is None:
        metadata = dict()
    elif not isinstance(metadata, dict):
        raise TypeError(
            f"The metadata of the run-function cannot be of type {type(metadata)}"
        )
    output["metadata"] = metadata

    # check if multiple observations returned
    objective = np.asarray(output["objective"])
    if objective.ndim == 2:
        output["objective"] = objective[1, -1].tolist()
        output["observations"] = objective.tolist()

    return output


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
    patience,
    raise_exception,
    func,
    log_interval=10,
    *args,
    **kwargs,
):
    """Decorator to use on a ``run_function`` to profile its execution-time and peak memory usage.

    Args:
        memory_limit (int): In bytes, if set to a positive integer, the memory usage is measured at regular intervals and the function is interrupted if the memory usage exceeds the limit. If set to ``-1``, only the peak memory is measured. If the executed function is busy outside of the Python interpretor, this mechanism will not work properly. Defaults to ``-1``.
        memory_tracing_interval (float): In seconds, the interval at which the memory usage is measured. Defaults to ``0.1``.
        patience: In seconds, the number of intervals after which a memory violation will result in a kill

    Returns:
        function: a decorated function.
    """

    if "logger" not in kwargs:
        logger = logging.getLogger("LCDB")
    else:
        logger = kwargs["logger"]

    timestamp_start = time.time()
    timestamp_last_log_message = -np.inf

    p = psutil.Process()  # get the current process

    output = None

    try:
        with ProcessPoolExecutor(max_workers=1) as executor:

            # trick to get the PID of the process that runs this job, to being able to kill it later
            future = executor.submit(os.getpid)
            pid = future.result()
            p = psutil.Process(pid)

            # submit actual job
            future = executor.submit(func, *args, **kwargs)

            memory_peak = p.memory_info().rss

            # start monitoring memory consumption of the process
            first_time_of_violation_in_sequence = None
            remaining_time_before_kill = None
            while not future.done():

                # in bytes (not the peak memory but last snapshot)
                memory_now = p.memory_info().rss
                memory_peak = max(memory_now, memory_peak)
                now = time.time()
                if now - timestamp_last_log_message > log_interval:
                    logger.info(f"Current memory consumption: {memory_now // 1024**2}MB ({np.round(100.0 * memory_now / memory_limit, 2)}% of the defined limit)")
                    timestamp_last_log_message = now
                

                if memory_limit > 0 and memory_now > memory_limit:
                    if first_time_of_violation_in_sequence is None:
                        first_time_of_violation_in_sequence = time.time()
                    elapsed_time_in_forbidden_zone = (time.time() - first_time_of_violation_in_sequence)
                    remaining_time_before_kill = patience - elapsed_time_in_forbidden_zone
                    if remaining_time_before_kill > 0:
                        logger.warning(f"Function is exceeding allowed memory ({memory_now // 1024**2}/{memory_limit // 1024**2}MB). Remaining time before kill: {round(remaining_time_before_kill, 1)}s.")
                    else:
                        output = "F_memory_limit_exceeded"
                        os.kill(pid, signal.SIGTERM)
                        future.cancel()

                        if raise_exception:
                            raise CancelledError(
                                f"Memory limit exceeded: {memory_peak} > {memory_limit}"
                            )
                        else:
                            logger.warning(
                                f"Function call was cancelled due to exceeded memory limit: {memory_peak} > {memory_limit}"
                            )

                        break
                
                else:
                    if first_time_of_violation_in_sequence is not None:
                        logger.info("Memory usage is back in the allowed region.")
                        first_time_of_violation_in_sequence = None
                        remaining_time_before_kill = None
                time.sleep(memory_tracing_interval)

            if output is None:
                try:
                    output = future.result()
                except Exception as e:
                    traceback_of_error = traceback.format_exc()
                    output = {"objective": "F", "metadata": {"traceback": str(traceback_of_error)}}
                    logger.exception(e)

    except BrokenExecutor:
        pass

    timestamp_end = time.time()

    output = standardize_run_function_output(output)
    metadata = {
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }

    metadata["memory_max"] = memory_peak

    metadata.update(output["metadata"])
    output["metadata"] = metadata

    return output


def get_schedule(name, **kwargs):
    """Get a schedule given its name and optional arguments.

    Args:
        name (str): name of the schedule.
        **kwargs: optional arguments to pass to the schedule.
    """
    if type(name) == int:
        return [name]
    
    if name == "full":
        return get_linear_schedule(**kwargs)
    elif name == "linear":
        return get_linear_schedule(**kwargs)
    elif name == "first":
        return get_schedule("power", **kwargs)[:1]
    elif name == "last":
        return [kwargs["max_anchor"]]
    elif name.startswith("power"):
        if not "max_anchor" in kwargs:
            raise ValueError(f"power schedule requires keyword `max_anchor`")
        if "-" in name:
            exploded_name = name.split("-")
            if len(exploded_name) != 4:
                raise ValueError(
                    "if schedule starts with 'power-' it must be in format 'power-<base>-<power>-<delay>"
                )
            kwargs["base"] = float(exploded_name[1])
            kwargs["power"] = float(exploded_name[2])
            kwargs["delay"] = int(exploded_name[3])
        return sorted(set(get_power_schedule(**kwargs)))
    else:
        try:
            return [int(name)]
        except:
            pass
        raise ValueError(f"Unknown schedule: {name}")


def get_linear_schedule(max_anchor: int, step: 1 = 1, **kwargs):
    return sorted(range(1, max_anchor + 1, step))


def get_power_schedule(max_anchor: int, base=2, power=0.5, delay: int = 7, **kwargs):
    """Get a power schedule up to anchor `max_anchor`."""
    anchors = []
    k = 1
    while True:
        exponent = (delay + k) * power
        sample_size = int(np.round(base**exponent))
        if sample_size > max_anchor:
            break
        anchors.append(sample_size)
        k += 1
    if len(anchors) > 0 and anchors[-1] < max_anchor:
        anchors.append(max_anchor)
    return sorted(anchors)


def decision_fun_to_proba(decision_fun_vals):
    """
    take a vector or matrix of decision function values and turn them into probabilities through a softmax

    :param decision_fun_vals:
    :return:
    """
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    if len(decision_fun_vals.shape) == 2:
        return softmax(decision_fun_vals, axis=1)
    else:  # if the decision function values is only a vector, then these are the probs of the positive class
        a = sigmoid(decision_fun_vals)
        return np.column_stack([1 - a, a])

def estimate_memory_consumption_for_dataset(shape, dtype=np.float64, unit="B"):

    if dtype == np.float64:
        memory_per_field = 8
    elif dtype == np.float16:
        memory_per_field = 2
    else:
        raise ValueError(f"Cannot estimate the memory consumption for an array of type {dtype}")

    # Compute memory usage
    num_elements = np.prod(shape)  # Total number of elements
    memory_bytes = num_elements * memory_per_field

    if unit == "B":
        pass
    elif unit == "KB":
        memory_bytes /= 1024
    elif unit == "MB":
        memory_bytes /= (1024**2)  # Convert to MB
    elif unit == "GB":
        memory_bytes /= (1024**3)  # Convert to GB
    else:
        raise ValueError(f"Unit must be 'B', 'KB', 'MB' or 'GB' but is {unit}")
    return memory_bytes

def convert_deephyper_result_row_to_dict(row):
    config = {}
    experiment = {}
    remaining_fields = {}
    for field, value in row.items():

        # store nans as null
        if type(value) == float and np.isnan(value):
            value = None

        if field.startswith('p:'):
            config[field[2:]] = value
        elif field == "m:json":
            remaining_fields["results"] = value
        elif field.startswith('m:'):
            if field.startswith("m:timestamp_") or field in ["m:lcdb_version"] or field in ["m:job_name"]:
                experiment[field[2:]] = value
            else:
                remaining_fields[field[2:]] = value
        elif field in ["job_id", "job_status"]:
            experiment[field] = value
        elif field.startswith('sol.'):
            pass
        elif field == "objective":
            if type(value) == str:
                remaining_fields["error"] = value
        else:
            remaining_fields[field] = value
    row_to_write = {"config": config, "experiment_metadata": experiment}
    row_to_write.update(remaining_fields)
    return row_to_write

def deephyper_results_to_jsonl(path_to_deephyper_results_csv, path_to_output_jsonl):

    df = pd.read_csv(path_to_deephyper_results_csv)

    with jsonlines.open(path_to_output_jsonl, mode='w') as writer:
        for _, row in df.iterrows():
            writer.write(convert_deephyper_result_row_to_dict(row))


class StatusFileManager:

    EXPERIMENT_STATUS_SUBMITTED = "submitted"  # has been given to the scheduler
    EXPERIMENT_STATUS_STARTED = "started"   # has been invoked by the scheduler
    EXPERIMENT_STATUS_RUNNING = "running"   # execution has been started at Python level
    EXPERIMENT_STATUS_COMPLETED= "completed"    # the results of all hyperparameter configurations are in

    VALID_STATI = [EXPERIMENT_STATUS_SUBMITTED,EXPERIMENT_STATUS_STARTED, EXPERIMENT_STATUS_RUNNING, EXPERIMENT_STATUS_COMPLETED]

    def __init__(self, working_directory="./exp-checkpoints"):
        self.working_directory = working_directory
    
    def get_path_to_status_file(self, workflow, openmlid, campaign, workflowseed, testseed, valseed, status):
        if status not in StatusFileManager.VALID_STATI:
            raise ValueError(f"status must be in {StatusFileManager.VALID_STATI} but is {status}")
        return pathlib.Path(f"{self.working_directory}/{campaign}/{workflow}-{openmlid}-{workflowseed}-{testseed}-{valseed}.{status}")

    def does_status_file_exist(self, workflow, openmlid, campaign, workflowseed, testseed, valseed, status):
        return self.get_path_to_status_file(workflow, openmlid, campaign, workflowseed, testseed, valseed, status).exists()

    def create_status_file(self, workflow, openmlid, campaign, workflowseed, testseed, valseed, status):
        path = self.get_path_to_status_file(workflow, openmlid, campaign, workflowseed, testseed, valseed, status)
        path.parent.mkdir(exist_ok=True, parents=True)
        if self.does_status_file_exist(workflow, openmlid, campaign, workflowseed, testseed, valseed, status):
            raise ValueError(f"status file already exists for {workflow=}, {openmlid=}, {campaign=}, {workflowseed=}, {testseed=}, {valseed=}, {status=}")
        path.touch()

    def remove_status_file(self, workflow, openmlid, campaign, workflowseed, testseed, valseed, status):
        pathlib.Path(self.get_path_to_status_file(workflow=workflow, openmlid=openmlid, campaign=campaign, workflowseed=workflowseed, testseed=testseed, valseed=valseed, status=status)).unlink(missing_ok=True)

def get_random_state(randomness_description):    
    if isinstance(randomness_description, np.random.RandomState):
        return randomness_description
    return np.random.RandomState(randomness_description)
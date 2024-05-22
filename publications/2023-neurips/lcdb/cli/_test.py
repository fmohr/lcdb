"""Command line to test workflow."""

import functools
import json
import os
import logging

import lcdb.json
from deephyper.evaluator import RunningJob
from lcdb.utils import import_attr_from_module, terminate_on_memory_exceeded

from ._run import run

# Avoid Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "test"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Run experiments with DeepHyper."
    )

    subparser.add_argument(
        "-id",
        "--openml-id",
        type=int,
        required=True,
        help="The identifier of the OpenML dataset.",
    )
    subparser.add_argument(
        "-w",
        "--workflow-class",
        type=str,
        required=True,
        help="The 'path' of the workflow to train.",
    )
    subparser.add_argument(
        "-tt",
        "--task-type",
        type=str,
        required=False,
        choices=["classification", "regression"],
        default="classification",
        help="The type of the supervised ML task. Either 'classification' or 'regression'.",
    )
    subparser.add_argument(
        "-m",
        "--monotonic",
        action="store_true",
        default=False,
        required=False,
        help="A boolean indicating if the sample-wise learning curve should be monotonic (i.e., sample set at smaller anchors are always included in sample sets at larger anchors) or not.",
    )
    subparser.add_argument(
        "-vs",
        "--valid-seed",
        type=int,
        default=42,
        required=False,
        help="Random state seed of train/validation split.",
    )
    subparser.add_argument(
        "-ts",
        "--test-seed",
        type=int,
        default=42,
        required=False,
        help="Random state seed of train+validation/test split.",
    )
    subparser.add_argument(
        "-ws",
        "--workflow-seed",
        type=int,
        default=42,
        required=False,
        help="Random state seed of the workflow.",
    )
    subparser.add_argument(
        "-vp",
        "--valid-prop",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of validation/(train+validation).",
    )
    subparser.add_argument(
        "-tp",
        "--test-prop",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of test/data.",
    )
    subparser.add_argument(
        "--timeout-on-fit",
        type=int,
        default=-1,
        required=False,
        help="Timeout in seconds for the fit method. Defaults to -1 for unlimited time.",
    )
    subparser.add_argument(
        "--parameters",
        type=str,
        default=None,
        required=False,
    )
    subparser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        required=False,
    )
    subparser.add_argument(
        "--anchor-schedule",
        default="power",
        type=str,
        help="The type of schedule for anchors (over samples of the dataset). Value in ['linear', 'last', 'power'].",
    )
    subparser.add_argument(
        "--epoch-schedule",
        default="power",
        type=str,
        help="The type of schedule for anchors (over learning iterations of the workflow). Value in ['linear', 'last', 'power'].",
    )
    subparser.set_defaults(func=function_to_call)


def main(
    openml_id,
    workflow_class,
    task_type,
    monotonic,
    valid_seed,
    test_seed,
    workflow_seed,
    valid_prop,
    test_prop,
    timeout_on_fit,
    parameters,
    verbose,
    anchor_schedule,
    epoch_schedule,
):
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

    # No parameters are given the default configuration is used
    if parameters is None:
        WorkflowClass = import_attr_from_module(workflow_class)
        config_space = WorkflowClass.config_space()
        config_default = dict(config_space.get_default_configuration())
    else:
        config_default = json.loads(parameters)

    # create controller
    if task_type not in ["classification", "regression"]:
        raise ValueError(
            f"Task type must be 'classification' or 'regression' but is {task_type}."
        )

    memory_limit_giga_bytes = float(
        os.environ.get("LCDB_EVALUATION_MEMORY_LIMIT", 10)
    )  # in GB

    # TODO: memory_limit should replaced and passed as a parameter
    memory_limit = memory_limit_giga_bytes * (1024**3)
    memory_tracing_interval = 0.1
    raise_exception = False
    run_function = functools.partial(
        terminate_on_memory_exceeded,
        memory_limit,
        memory_tracing_interval,
        raise_exception,
        run,
    )

    output = run_function(
        RunningJob(id=0, parameters=config_default),
        openml_id=openml_id,
        workflow_class=workflow_class,
        task_type=task_type,
        monotonic=monotonic,
        valid_seed=valid_seed,
        test_seed=test_seed,
        workflow_seed=workflow_seed,
        valid_prop=valid_prop,
        test_prop=test_prop,
        timeout_on_fit=timeout_on_fit,
        anchor_schedule=anchor_schedule,
        epoch_schedule=epoch_schedule,
    )

    # check that the output can indeed be compiled into a string using JSON
    out = lcdb.json.dumps(output, indent=2)
    print(out)

    traceback = output["metadata"].get("traceback")
    if traceback is not None and len(traceback) > 0:
        print(traceback[1:-1])

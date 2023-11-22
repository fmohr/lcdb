"""Command line to test workflow."""
import logging
import os
import pathlib

import lcdb.json
import pandas as pd
from deephyper.evaluator import Evaluator, RunningJob
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from lcdb.utils import import_attr_from_module
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
    subparser.set_defaults(func=function_to_call)


def main(
    openml_id,
    workflow_class,
    monotonic,
    valid_seed,
    test_seed,
    workflow_seed,
    valid_prop,
    test_prop,
    timeout_on_fit,
):
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()
    config_default = dict(config_space.get_default_configuration())

    output = run(
        RunningJob(id=0, parameters=config_default),
        openml_id=openml_id,
        workflow_class=workflow_class,
        monotonic=monotonic,
        valid_seed=valid_seed,
        test_seed=test_seed,
        workflow_seed=workflow_seed,
        valid_prop=valid_prop,
        test_prop=test_prop,
        timeout_on_fit=timeout_on_fit,
        raise_errors=True,
    )

    # check that the output can indeed be compiled into a string using JSON
    out = lcdb.json.dumps(output, indent=2)
    print(out)

    traceback = output["metadata"].get("traceback")
    if traceback is not None and len(traceback) > 0:
        print(traceback[1:-1])
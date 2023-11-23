"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import logging
import os
import pathlib

import numpy as np
import pandas as pd
from lcdb.data import load_task


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "fetch"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Print information about the task."
    )

    subparser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="The task ID. For example, the task 61 from OpenML will have task ID 'openml.61'",
    )

    subparser.set_defaults(func=function_to_call)


def main(
    task_id: str,
):
    """Entry point for the command line interface."""

    print(f"Loading task '{task_id}'...", end="")
    (X, y), dataset_metadata = load_task(task_id)
    print(" done!")

    print(f" * X shape: {np.shape(X)}")
    print(f" * y shape: {np.shape(y)}")
    print(f" * Classes: {dataset_metadata['num_classes']}")

    for k,v in dataset_metadata.items():
        print(f" * {k.capitalize()}: {v}")

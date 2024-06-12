"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import numpy as np
from ..data import load_task
from ..data.split import train_valid_test_split


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
    subparser.add_argument(
        "-vp",
        "--valid-prop",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of validation/(train+validation+test).",
    )
    subparser.add_argument(
        "-tp",
        "--test-prop",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of test/(train+validation+test).",
    )

    subparser.set_defaults(func=function_to_call)


def main(
    task_id: str,
    valid_prop: float,
    test_prop: float,
):
    """Entry point for the command line interface."""

    print(f"Loading task '{task_id}'...", end="")
    (X, y), dataset_metadata = load_task(task_id)
    print(" done!\n")

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = train_valid_test_split(
        X,
        y,
        test_seed=42,
        valid_seed=42,
        test_prop=test_prop,
        valid_prop=valid_prop,
        stratify=True,
    )

    print(f" * Type   : {dataset_metadata['type']}")
    print(f" * X shape: {np.shape(X)}")
    print(f" * y shape: {np.shape(y)}")
    print(f" * Classes: {dataset_metadata['num_classes']}")
    print(f" * Categories: {dataset_metadata['categories']}")

    print()
    print(" --- Splitting ---")

    print(f" * Train X shape: {np.shape(X_train)}")
    print(f" * Train y shape: {np.shape(y_train)}")
    print(f" * Valid X shape: {np.shape(X_valid)}")
    print(f" * Valid y shape: {np.shape(y_valid)}")
    print(f" * Test  X shape: {np.shape(X_test)}")
    print(f" * Test  y shape: {np.shape(y_test)}")

    print()
    print(" --- Description ---")
    print(dataset_metadata["description"])

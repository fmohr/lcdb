"""Command line to create a list of hyperparameter configurations to be evaluated later."""
from ..builder.utils import import_attr_from_module


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "space"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Print the hyperparameter space."
    )

    subparser.add_argument("-w", "--workflow-class", type=str, required=True)

    subparser.set_defaults(func=function_to_call)


def main(
    workflow_class,
):
    """Entry point for the command line interface."""

    # Load the workflow to get its config space
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()

    print(config_space)

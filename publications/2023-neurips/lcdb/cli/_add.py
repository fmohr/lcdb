"""Command line to add experiment results to the repository."""

import os
import pathlib

# Avoid Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "add"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Run sets of experiments in the LCDB 2.0 layout."
    )

    subparser.add_argument(
        "-r",
        "--repository",
        type=str,
        required=False,
        default=None,
        help="The repository to which the results should be added.",
    )

    subparser.add_argument(
        "-c",
        "--campaign",
        type=str,
        required=True,
        help="The folder where workflow configs reside and results have been stored.",
    )

    subparser.add_argument('result_files', nargs='+', help='result files')
    subparser.set_defaults(func=function_to_call)


def main(**kwargs):

    """Entry point for the command line interface."""
    from ..db import LCDB

    lcdb = LCDB()
    repository_name = kwargs["repository"] if kwargs["repository"] is not None else list(lcdb.repositories.keys())[0]

    r = lcdb.repositories[repository_name]

    r.add_results(
        kwargs["campaign"],
        *kwargs["result_files"]
    )

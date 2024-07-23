"""Command line to initialize an LCDB in the current folder."""

import os

# Avoid Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "init"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Run sets of experiments in the LCDB 2.0 layout."
    )
    subparser.set_defaults(func=function_to_call)


def main():

    """Entry point for the command line interface."""
    from ..db import init_database
    init_database()

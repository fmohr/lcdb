"""Command line to initialize an LCDB in the current folder (or another given one)."""

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
        subparser_name, help="To initialize a new LCDB config."
    )

    subparser.add_argument(
        'path',
        nargs='?',
        default=None,
        help='The path where to initialize the LCDB.'
    )

    subparser.set_defaults(func=function_to_call)


def main(path=None):

    """Entry point for the command line interface."""
    from ..db import LCDB

    if path is None:
        path = os.getcwd()

    lcdb = LCDB(path=path)
    if lcdb.exists():
        print(f"An LCDB is already initialized at path {path}.")
    else:
        lcdb.create()

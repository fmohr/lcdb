"""Command line to create summaries of data availability and experiment meta-data (memory and time consumptions) from LCDB 2.0 repositories"""
import pandas as pd


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "stats"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Summarizes result availability contained LCDB 2.0 repositories."
    )

    subparser.add_argument(
        "-w",
        "--workflow-class",
        type=str,
        required=False,
        help="The 'path' of the workflow to train.",
    )

    subparser.add_argument(
        "-id",
        "--openml-id",
        type=int,
        required=False,
        help="The identifier of the OpenML dataset.",
    )

    subparser.add_argument(
        "-r",
        "--repositories",
        type=str,
        required=False,
        help="comma separated paths to repository folders to use.",
    )

    subparser.set_defaults(func=function_to_call)


def main(
    workflow_class,
    openml_id,
    repositories
):

    from ..db import LCDB

    lcdb = LCDB()
    
    ## @Andreas: please add your logic here.
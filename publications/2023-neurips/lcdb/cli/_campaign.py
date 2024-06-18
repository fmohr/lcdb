"""Command line to run experiments."""

import os
from ._utils import parse_comma_separated_strs, parse_comma_separated_ints
import logging

# Avoid Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "campaign"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Run sets of experiments in the LCDB 2.0 layout."
    )

    subparser.add_argument(
        "-c",
        "--campaign-dir",
        type=str,
        required=True,
        help="The folder where workflow configs reside and results will be stored.",
    )

    subparser.add_argument(
        "-id",
        "--openml-ids",
        type=parse_comma_separated_ints,
        required=True,
        help="The identifiers of the OpenML datasets.",
    )

    subparser.add_argument(
        "-w",
        "--workflow-classes",
        type=parse_comma_separated_strs,
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
        "--valid-seeds",
        type=parse_comma_separated_ints,
        default=[42],
        required=False,
        help="comma separated random seeds of train/validation split.",
    )
    subparser.add_argument(
        "-ts",
        "--test-seeds",
        type=parse_comma_separated_ints,
        default=[42],
        required=False,
        help="comma separated random seeds of train+validation/test split.",
    )
    subparser.add_argument(
        "-ws",
        "--workflow-seeds",
        type=parse_comma_separated_ints,
        default=[42],
        required=False,
        help="comma separated random seeds of the workflow.",
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
    subparser.add_argument(
        "--timeout-on-fit",
        type=int,
        default=-1,
        required=False,
        help="Timeout in seconds for the fit method. Defaults to -1 for unlimited time.",
    )
    subparser.add_argument(
        "-d",
        "--log-dir",
        type=str,
        default=".",
        required=False,
        help="Directory where to store the outputs/logs.",
    )
    subparser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=1800,
        required=False,
        help="Overall timeout in seconds.",
    )
    subparser.add_argument(
        "--workflow-memory-limit",
        type=float,
        default=1750.0,
        required=False,
        help="Memory limit per config (MBs).",
    )
    subparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        required=False,
        help="Boolean to activate or not the verbose mode.",
    )
    subparser.add_argument(
        "-e",
        "--evaluator",
        default="ray",
        type=str,
        help="The evaluator to use. It can be 'serial', 'thread', 'process', 'ray' or 'mpicomm'.",
    )
    subparser.add_argument(
        "--num-workers",
        default=-1,
        type=int,
        help="The number of workers to use with the evaluator. Defaults to -1 for all available workers.",
    )
    subparser.add_argument(
        "--anchor-schedule",
        default="power",
        type=str,
        help="The type of schedule for anchors (over samples of the dataset). Value in ['linear', 'last', 'power'].",
    )
    subparser.add_argument(
        "--epoch-schedule",
        default="full",
        type=str,
        help="The type of schedule for anchors (over learning iterations of the workflow). Value in ['linear', 'last', 'power'].",
    )
    subparser.set_defaults(func=function_to_call)


def main(verbose, **kwargs):

    """Entry point for the command line interface."""
    from ..experiments._experiments import LCDB  # lazy import to avoid slow down of the CLI

    # define stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger = logging.getLogger("LCDB")
    kwargs["logger"] = logger
    if verbose:

        logger.setLevel(logging.DEBUG)
        logger.addHandler(ch)


    LCDB.run_campaign(verbose=verbose, **kwargs)

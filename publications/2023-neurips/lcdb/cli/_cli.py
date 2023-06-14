"""
$ lcdb --help
$ lcdb create ...
$ lcdb run ...
"""
import argparse

from lcdb.cli import _create, _run


def create_parser():
    """
    :meta private:
    """
    parser = argparse.ArgumentParser(description="LCDB command line.")

    subparsers = parser.add_subparsers()

    # creation of experiments
    _create.add_subparser(subparsers)

    # execution of experiments
    _run.add_subparser(subparsers)

    return parser


def main():
    """
    :meta private:
    """
    parser = create_parser()

    args = parser.parse_args()

    if hasattr(args, "func"):
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)
    else:
        parser.print_help()
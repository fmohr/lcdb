"""
$ lcdb --help
$ lcdb create ...
$ lcdb run ...
"""
import argparse

from . import _space, _create, _fetch, _results, _run, _test, _campaign, _plot


def create_parser():
    """
    :meta private:
    """
    parser = argparse.ArgumentParser(description="LCDB command line.")

    subparsers = parser.add_subparsers()

    # print the hyperparameter space

    _space.add_subparser(subparsers)

    # generation of hyperparameter configurations
    _create.add_subparser(subparsers)

    # fetch data
    _fetch.add_subparser(subparsers)

    # test workflow
    _test.add_subparser(subparsers)

    # execution of experiments
    _run.add_subparser(subparsers)

    # execution of experiments
    _campaign.add_subparser(subparsers)

    # plot results
    _results.add_subparser(subparsers)

    # plot results
    _plot.add_subparser(subparsers)

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


if __name__ == "__main__":
    main()

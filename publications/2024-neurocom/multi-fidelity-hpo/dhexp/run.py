import warnings

warnings.simplefilter("ignore")

import argparse
import importlib
import json
import pathlib
import sys

from dhexp.utils import import_attr_from_module


def create_parser():
    parser = argparse.ArgumentParser(description="Command line to run experiments.")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Problem on which to experiment.",
    )
    parser.add_argument(
        "--search",
        type=str,
        required=True,
        help="Search the experiment must be done with.",
    )
    parser.add_argument(
        "--search-kwargs",
        type=str,
        default="{}",
        required=False,
    )
    parser.add_argument(
        "--stopper",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--stopper-kwargs",
        type=str,
        default="{}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Search maximum duration (in min.) for each optimization.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=-1,
        help="Number of iterations to run for each optimization.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Wether to activate or not the verbose mode.",
    )
    return parser


def main(args):
    args = vars(args)

    # load the problem
    pmodule = importlib.import_module(args["problem"])

    # Set up the stopper (argument of the search)
    stopper_class = import_attr_from_module(args["stopper"])
    stopper_kwargs = json.loads(args["stopper_kwargs"].replace("'", '"'))
    Stopper = stopper_class(**stopper_kwargs)

    # Set up the search
    search_class = import_attr_from_module(args["search"])
    search_kwargs = json.loads(args["search_kwargs"].replace("'", '"'))
    search_kwargs["problem"] = pmodule.problem
    search_kwargs["evaluator"] = pmodule.run
    search_kwargs["stopper"] = Stopper
    search_kwargs["verbose"] = args["verbose"]

    pathlib.Path(search_kwargs.get("log_dir", ".")).mkdir(parents=True, exist_ok=True)

    Search = search_class(**search_kwargs)
    results = Search.search(max_evals=args["max_evals"], timeout=args["timeout"])


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    # delete arguments to avoid conflicts
    sys.argv = [sys.argv[0]]

    main(args)

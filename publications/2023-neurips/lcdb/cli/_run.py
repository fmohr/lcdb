"""Command line to run experiments."""


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "run"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Run experiments."
    )

    subparser.set_defaults(func=function_to_call)


def main(config: str, *args, **kwargs):
    """
    :meta private:
    """
    pass

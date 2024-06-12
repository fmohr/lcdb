"""Command line to fetch and aggregate results from LCDB 2.0 repositories"""


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "results"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Fetch and aggregate results from local or remote LCDB 2.0 repositories."
    )

    subparser.add_argument(
        "-w",
        "--workflow-class",
        type=str,
        required=True,
        help="The 'path' of the workflow to train.",
    )

    subparser.add_argument(
        "-id",
        "--openml-id",
        type=int,
        required=True,
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

    from ..analysis.results import ResultAggregator  # lazy import
    agg = ResultAggregator(
        repos=[s.strip().rstrip("/") for s in repositories.split(",")]
    )
    df = agg.get_results_for_all_configs(
        workflow=workflow_class,
        openmlid=openml_id
    )

    print(df)

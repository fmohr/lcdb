"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import os
import matplotlib.pyplot as plt
import pandas as pd
import json

from ._utils import parse_comma_separated_strs


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "plot"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Print the hyperparameter space."
    )

    subparser.add_argument(
        "-r",
        "--repositories",
        type=parse_comma_separated_strs,
        required=False,
        default=None,
        help="comma separated paths to repository folders to use.",
    )


    subparser.add_argument(
        "-w",
        "--workflow-classes",
        type=parse_comma_separated_strs,
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
        "-a",
        "--anchor",
        type=int,
        required=False,
        default=-1,
        help="sample-wise anchor (index); -1 for last.",
    )

    subparser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=False,
        default="brier_score",
        help="Metric for which to show performances.",
    )

    subparser.add_argument("-o", "--output-path", type=str, default="plot.png", required=False)
    
    subparser.add_argument("-t", "--plot-type", type=str, default="observation-wise")

    subparser.set_defaults(func=function_to_call)


def main(
    repositories,
    workflow_classes,
    openml_id,
    output_path,
    plot_type,
    anchor,
    metric
):

    from ..db import LCDB
    from ..analysis.plot import (
        plot_learning_curves,
        plot_observation_curves,
        plot_iteration_curves_dataset
    )

    df = LCDB().get_results(repositories=repositories, workflows=workflow_classes, openmlids=[openml_id])
    if df is not None:
        for workflow_class, df_results in df.groupby("m:workflow"):
            df_results = df_results[df_results["m:traceback"].isna()]

            if plot_type == "observation-wise":
                fig, ax = plot_observation_curves(df_results)
                fig.suptitle(f"Performance of {workflow_class} on {openml_id}")
                plt.draw()
            elif plot_type == "iteration-wise":
                fig, ax = plot_iteration_curves_dataset(df_results, metric=metric, sample_anchor=anchor)
                fig.suptitle(f"Performance of {workflow_class} on {openml_id}")
                plt.draw()
            else:
                raise ValueError(f"Unsupported plot type {plot_type}")

        plt.show()
    else:
        print("No data found in the database that could be plotted.")

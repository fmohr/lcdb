"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import os
import matplotlib.pyplot as plt
import numpy as np
import json


from ..analysis.json import QueryAnchorValues
from ..analysis.json import QueryMetricValuesFromAnchors
from ..analysis.score import balanced_accuracy_from_confusion_matrix

from ._utils import parse_comma_separated_strs


def save_stats(r_df, r_df_failed, output_path):

    csv_file = os.path.join(os.path.dirname(output_path), "statistics.json")

    dataset_id = int(r_df['m:openmlid'][0])

    # Calculate several memory statistics
    avg_mem_usage = r_df['m:memory'].mean()/(1024**3)
    std_mem_usage = r_df['m:memory'].std()/(1024**3)
    min_mem_usage = r_df['m:memory'].min()/(1024**3)
    max_mem_usage = r_df['m:memory'].max()/(1024**3)
    quartile_25_mem = r_df['m:memory'].quantile(0.25)/(1024**3)
    quartile_75_mem = r_df['m:memory'].quantile(0.75)/(1024**3)
    start_timestamp =  r_df['m:timestamp_start'].sum()
    end_timestamp = r_df['m:timestamp_end'].sum()
    # time elapsed for whole dataset
    time_elapsed = end_timestamp - start_timestamp


    # Calculate suggested number of cores and memory per core
    total_memory_per_node = 224
    dynamic_k = 1.0 + (std_mem_usage / avg_mem_usage)
    safe_memory_per_core = avg_mem_usage + dynamic_k * std_mem_usage
    suggested_core_usage = min(128, total_memory_per_node//safe_memory_per_core)
    memory_per_core = total_memory_per_node / suggested_core_usage

    # percentage of failed configs
    failed_perc = len(r_df_failed) / (len(r_df) + len(r_df_failed))*100

    stats = {
        'openML_id': dataset_id,
        'avg_mem_usage': avg_mem_usage,
        'std_mem_usage': std_mem_usage,
        'min_mem_usage': min_mem_usage,
        'max_mem_usage': max_mem_usage,
        'quantile_25_mem': quartile_25_mem,
        'quantile_75_mem': quartile_75_mem,
        'time_elapsed': time_elapsed,
        'cores_suggestion': suggested_core_usage,
        'mem_per_core': memory_per_core,
        'failed_mem(%)': failed_perc
    }

    with open(csv_file, 'w') as json_file:
        json.dump(stats, json_file, indent=4)


def plot_error_rate_vs_samples(r_df, output_path):

    """
    Method to plot and save Validation Balanced Error Rate vs. Number of Samples
    Input:
        - r_df: dataframe with successful job_id
        - output_path: output directory
    """
    l = []
    for hp_config, df_hp_config in r_df.groupby("p:C"):#["p:dropout_rate"]):
        source = df_hp_config["m:json"]
        query_anchor_values = QueryAnchorValues()
        anchor_values = source.apply(query_anchor_values).to_list()

        query_confusion_matrix_values = QueryMetricValuesFromAnchors("confusion_matrix", split_name="val")
        out = source.apply(query_confusion_matrix_values)
        #print(out)
        balanced_error_rate_values = np.array(out.apply(lambda x: list(map(lambda x: 1 - balanced_accuracy_from_confusion_matrix(x), x))).to_list())
        l.append(balanced_error_rate_values)
    balanced_error_rate_values = np.array(l)
    print(anchor_values, balanced_error_rate_values)

    for i, (xi, yi) in enumerate(zip(anchor_values, l)):
        anchor_values[i] = xi[:len(yi)]

    fig, ax = plt.subplots(figsize=figsize)
    plot_learning_curves(anchor_values, balanced_error_rate_values, metric_value_baseline=balanced_error_rate_values[0][-1], ax=ax)
    ax.axhline(y=balanced_error_rate_values[0].mean(axis=0)[-1], color="lime", linestyle="--")
    ax.set_xlabel(f"Number of Samples")
    ax.set_ylabel(f"Validation Balanced Error Rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig(os.path.join(os.path.dirname(output_path), "val_balanced_error_rate_vs_samples.jpg"), dpi=300, bbox_inches="tight")


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
        default=".",
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

    from ..analysis.results import ResultAggregator
    from ..analysis.plot import (
        plot_learning_curves,
        plot_observation_curves,
        plot_iteration_curves_dataset
    )

    results_aggregator = ResultAggregator(repositories)

    for workflow_class in workflow_classes:
        df_results = results_aggregator.get_results_for_all_configs(
            workflow=workflow_class,
            openmlid=openml_id
        )
        df_results = df_results[df_results["m:traceback"].isna()]

        with open("out.json", "w") as f:
            json.dump(df_results.iloc[0]["m:json"], f, indent=4)

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

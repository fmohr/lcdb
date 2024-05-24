"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import logging
import os
import gzip
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import json


from deephyper.analysis import rank
from deephyper.analysis._matplotlib import figure_size, update_matplotlib_rc

from lcdb.analysis import read_csv_results
from lcdb.analysis.json import QueryAnchorValues
from lcdb.analysis.json import QueryMetricValuesFromAnchors
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix

update_matplotlib_rc()
figsize = figure_size(252 * 1.8, 1.0)


def plot_learning_curves(
    fidelity_values,
    metric_values,
    mode="min",
    rank_method="ordinal",
    decimals=5,
    alpha=1.0,
    metric_value_baseline=None,
    plot_worse_than_baseline=True,
    ax=None,
    cmap=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # corresponds to iloc indexing
    if metric_value_baseline is not None:
        metric_max_fidelity = np.asarray(
            [metric_value_baseline] + [y[-1] for y in metric_values]
        )
    else:
        metric_max_fidelity = np.asarray([y[-1] for y in metric_values])

    if mode == "max":
        ranking = rank(-metric_max_fidelity, decimals=decimals, method=rank_method)
    elif mode == "min":
        ranking = rank(metric_max_fidelity, decimals=decimals, method=rank_method)
    else:
        raise ValueError(f"Unknown mode '{mode}' should be 'max' or 'min'.")

    if metric_value_baseline is not None and cmap is None:
        ranking_baseline = ranking[0]
        ranking = ranking[1:]

        center = ranking_baseline / max(ranking)
        q1 = center / 2
        q3 = center + (1 - center) / 2

        cmap = LinearSegmentedColormap.from_list(
            "custom",
            (
                # Edit this gradient at https://eltos.github.io/gradient/#0:00D0FF-25:0000FF-75:FF0000-100:FFD800
                (0.000, (0.000, 0.816, 1.000)),
                (q1, (0.000, 0.000, 1.000)),
                (q3, (1.000, 0.000, 0.000)),
                (1.000, (1.000, 0.847, 0.000)),
            ),
        )

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "custom",
            (
                # Edit this gradient at https://eltos.github.io/gradient/#0:00D0FF-25:0000FF-75:FF0000-100:FFD800
                (0.000, (0.000, 0.816, 1.000)),
                (0.250, (0.000, 0.000, 1.000)),
                (0.750, (1.000, 0.000, 0.000)),
                (1.000, (1.000, 0.847, 0.000)),
            ),
        )
    elif isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    else:
        cmap = cmap

    ranking_max = ranking.max()
    for i, (x, y) in enumerate(zip(fidelity_values, metric_values)):
        if not plot_worse_than_baseline:
            # if mode == "min" and metric_value_baseline and all(map(lambda yi: yi > metric_value_baseline , y)):
            if mode == "min" and metric_value_baseline is not None and y[-1] > metric_value_baseline:
                continue
            elif mode == "max" and metric_value_baseline is not None and -y[-1] > metric_value_baseline:
                continue
        ax.plot(x, y, color=cmap(ranking[i] / ranking_max), alpha=alpha)

    ax.grid()

    norm = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
    norm.set_clim(1, ranking_max)
    cb = plt.colorbar(norm, ax=plt.gca(), label="Rank")
    if metric_value_baseline is not None:
        cb.ax.axhline(ranking_baseline, c="lime", linewidth=2, linestyle="--")
    # plt.xlim(0, fidelities.max())

    return fig, ax


def pad_with_last(x, max_len):
    if len(x) < max_len:
        return x + [x[-1]] * (max_len - len(x))
    else:
        return x


def plot_regret_from_topk(fidelity_values, metric_values, topk=10, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    max_len = max(map(len, metric_values))

    # The x values to plot
    for x in fidelity_values:
        if len(x) == max_len:
            break

    metric_values = list(map(lambda x: pad_with_last(x, max_len), metric_values))
    metric_values = np.asarray(metric_values)

    # The best score at the maximum fidelity
    y_star = metric_values[:, -1].min(axis=0)

    # Compute the regrets
    idx_selected = map(lambda x: np.argpartition(x, kth=topk)[:topk], metric_values.T)
    regrets = list(map(lambda idx: (metric_values[idx, -1] - y_star), idx_selected))
    regrets_median = np.median(regrets, axis=1)
    regrets_min = np.quantile(regrets, q=0.1, axis=1)
    regrets_max = np.quantile(regrets, q=0.9, axis=1)

    ax.plot(x, regrets_median)
    ax.fill_between(x, regrets_min, regrets_max, alpha=0.2)
    ax.grid()

    return fig, ax

def plto_mem_per_job(r_df, output_path):
    """
    Method to plot and save peak memory usage per (successful) job id 
    Input:
        - r_df: dataframe with successful job_id 
        - output_path: output directory
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(r_df['job_id'], r_df['m:memory']/(1024**3), color='blue')

    avg_mem_usage = r_df['m:memory'].mean()/(1024**3)
    ax.axhline(y=avg_mem_usage, color='green', linestyle='--', linewidth=2, label=f'avereg_mem_use')

    ax.set_title('Memory Usage per Job ID')
    ax.set_xlabel('Job ID')
    ax.set_ylabel('Memory Peak (GBs)')
    plt.savefig(os.path.join(os.path.dirname(output_path), "jobID_vs_mem.jpg"), dpi=300, bbox_inches="tight")

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
    source = r_df["m:json"]
    query_anchor_values = QueryAnchorValues()
    anchor_values = source.apply(query_anchor_values).to_list()

    query_confusion_matrix_values = QueryMetricValuesFromAnchors("confusion_matrix", split_name="val")
    out = source.apply(query_confusion_matrix_values)
    balanced_error_rate_values = out.apply(lambda x: list(map(lambda x: 1 - balanced_accuracy_from_confusion_matrix(x), x))).to_list()

    for i, (xi, yi) in enumerate(zip(anchor_values, balanced_error_rate_values)):
        anchor_values[i] = xi[:len(yi)]

    fig, ax = plt.subplots(figsize=figsize)
    plot_learning_curves(anchor_values, balanced_error_rate_values, metric_value_baseline=balanced_error_rate_values[0][-1], ax=ax)
    ax.axhline(y=balanced_error_rate_values[0][-1], color="lime", linestyle="--")
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

    subparser.add_argument("-p", "--results-path", type=str, required=True)

    subparser.add_argument("-o", "--output-path", type=str, default="plot.png", required=True)
    
    subparser.add_argument("-t", "--plot-type", type=str, required=True)

    subparser.set_defaults(func=function_to_call)


def main(
    results_path,
    output_path,
    plot_type
):
    source_csv = f"{results_path}/results.csv.gz"
    """Entry point for the command line interface."""
    # Load a dataframe with the results
    # The dataframe is sorted by `job_id` (increasing) as parallel jobs scheduled asynchronously 
    # may be collected in a different order than when they were submitted.
    with gzip.GzipFile(source_csv, "rb") as f:        
        r_df, r_df_failed = read_csv_results(f)

    # Plot: Validation Balanced Error Rate vs. Number of Samples
    plot_error_rate_vs_samples(r_df, output_path)
    # Plot: Memory usage per job ID
    plto_mem_per_job(r_df, output_path)
    # Save stats for current execution
    save_stats(r_df, r_df_failed, output_path)

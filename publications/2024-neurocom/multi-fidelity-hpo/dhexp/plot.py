import argparse
import inspect
import os
import pathlib

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.
    
    From: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

width, height = set_size(252, fraction=1.0)
print(f"{width=}, {height=}")

font = 9

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.size': font,
    'figure.figsize': (width, height), 
    'figure.facecolor': 'white', 
    'savefig.dpi': 300, 
    'figure.subplot.bottom': 0.125, 
    'figure.edgecolor': 'white',
    'xtick.labelsize': font,
    'ytick.labelsize': font,
    'legend.fontsize': font,
})


HERE = os.path.dirname(os.path.abspath(__file__))
FILE_EXTENSION = "pdf"
PRINT_TITLE = False
MODE = "max"


def yaml_load(path):
    with open(path, "r") as f:
        yaml_data = yaml.load(f, Loader=Loader)
    return yaml_data


def yaml_dump(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=Dumper)


def rename_column(df):
    columns = {
        "m:timestamp_start": "timestamp_start",
        "m:timestamp_end": "timestamp_end",
    }
    df = df.rename(columns=columns)
    return df


def load_results(exp_root: str, exp_config: dict) -> dict:
    data = {}
    for exp_prefix in exp_config["data"]:
        if "rep" in exp_config["data"][exp_prefix]:
            dfs = []
            for rep in exp_config["data"][exp_prefix].get("rep"):
                exp_results_path = os.path.join(
                    exp_root, f"{exp_prefix}-{rep}/results.csv"
                )
                print(f"Loading {exp_results_path}")
                df = pd.read_csv(exp_results_path, index_col=0)
                df = df.dropna().reset_index(drop=True)
                df = rename_column(df)
                df = df.astype({"objective": "float64"})

                df["timestamp_end"] = df["timestamp_end"] - df.loc[0, "timestamp_start"]
                df["timestamp_start"] = df["timestamp_start"] - df.loc[0, "timestamp_start"]
                df["duration"] = df["timestamp_end"] - df["timestamp_start"]

                dfs.append(df)
                data[exp_prefix] = dfs
        else:
            exp_results_path = os.path.join(exp_root, f"{exp_prefix}/results.csv")
            df = pd.read_csv(exp_results_path, index_col=0)
            subdf = df.dropna().reset_index(drop=True)

            print(f"Rows dropped with NA: {(len(df) - len(subdf))/len(df)*100}%")
            df = rename_column(subdf)
            df = df.astype({"objective": "float64"})

            df["timestamp_end"] = df["timestamp_end"] - df.loc[0, "timestamp_start"]
            df["timestamp_start"] = df["timestamp_start"] - df.loc[0, "timestamp_start"]
            df["duration"] = df["timestamp_end"] - df["timestamp_start"]
            data[exp_prefix] = df
    return data


@ticker.FuncFormatter
def hour_major_formatter(x, pos):
    x = float(f"{x/3600:.2f}")
    if x % 1 == 0:
        x = str(int(x))
    else:
        x = f"{x:.2f}"
    return x


@ticker.FuncFormatter
def minute_major_formatter(x, pos):
    x = float(f"{x/60:.2f}")
    if x % 1 == 0:
        x = str(int(x))
    else:
        x = f"{x:.2f}"
    return x


def plot_scatter_multi(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            for i, exp_df in enumerate(exp_dfs):
                x, y = exp_df.timestamp_end.to_numpy(), -exp_df.objective.to_numpy()

                plt_kwargs = dict(
                    color=exp_config["data"][exp_name]["color"], s=2, alpha=0.5
                )
                if i == 0:
                    plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

                plt.scatter(x, y, **plt_kwargs)
        else:
            x, y = exp_df.timestamp_end.to_numpy(), exp_df.objective.to_numpy()

            x = x + 1
            y = exp_config.get("best_objective", 1) - y

            plt.scatter(
                x,
                y,
                color=exp_config["data"][exp_name]["color"],
                label=exp_config["data"][exp_name]["label"],
                s=1,
                alpha=0.2,
            )

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    lgnd = plt.legend()
    for i in range(len(lgnd.legend_handles)):
        lgnd.legend_handles[i]._sizes = [10]
        lgnd.legend_handles[i]._alpha = 1

    plt.ylabel(exp_config.get("ylabel", "Objective"))
    plt.xlabel("Search time (min.)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def plot_scatter_multi_iter(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            for i, exp_df in enumerate(exp_dfs):
                exp_df = exp_df.sort_values("timestamp_end")
                x, y = (
                    np.arange(len(exp_df)),
                    exp_df.objective.to_numpy(),
                )

                y = -y if MODE == "min" else y

                plt_kwargs = dict(
                    color=exp_config["data"][exp_name]["color"], s=10, alpha=0.5
                )
                if i == 0:
                    plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

                plt.scatter(x, y, **plt_kwargs)
        else:
            exp_df = exp_df.sort_values("timestamp_end")
            x, y = (
                np.arange(len(exp_df)),
                exp_df.objective.to_numpy(),
            )

            y = -y if MODE == "min" else y

            plt.scatter(
                x,
                y,
                color=exp_config["data"][exp_name]["color"],
                label=exp_config["data"][exp_name]["label"],
                s=10,
                alpha=0.5,
            )

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel(exp_config.get("ylabel", "Objective"))
    plt.xlabel("#Evaluations")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0)

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def plot_scatter_multi_budget(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    default_budget = exp_config.get("default_budget", 1)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            for i, exp_df in enumerate(exp_dfs):
                exp_df = exp_df.sort_values("timestamp_end")

                if "m:budget" not in exp_df.columns:
                    exp_df["m:budget"] = default_budget
                x = exp_df["m:budget"].cumsum()

                y = exp_df.objective.to_numpy()

                y = -y if MODE == "min" else y

                plt_kwargs = dict(
                    color=exp_config["data"][exp_name]["color"], s=10, alpha=0.5
                )
                if i == 0:
                    plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

                plt.scatter(x, y, **plt_kwargs)
        else:

            exp_df = exp_df.sort_values("timestamp_end")

            if "m:budget" not in exp_df.columns:
                exp_df["m:budget"] = default_budget
            x = exp_df["m:budget"].cumsum()

            y = exp_df.objective.to_numpy()

            y = -y if MODE == "min" else y

            plt.scatter(
                x,
                y,
                color=exp_config["data"][exp_name]["color"],
                label=exp_config["data"][exp_name]["label"],
                s=10,
                alpha=0.5,
            )

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel(exp_config.get("ylabel", "Objective"))
    plt.xlabel("Budget")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    # if exp_config.get("xlim"):
    #     plt.xlim(*exp_config.get("xlim"))
    # else:
    # plt.xlim(0)

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    if exp_config.get("xscale"):
        plt.xscale(exp_config.get("xscale"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def count_better(df, baseline_perf):

    hp_names = [cname for cname in df.columns if "p:" in cname]
    df_str = df.astype({k: str for k in hp_names})
    df_str["hash"] = df_str[hp_names].agg("-".join, axis=1)
    df_str = df_str[["hash", "objective", "timestamp_end"]]

    earliest_timestamps = (
        df_str.groupby("hash")
        .agg({"timestamp_end": "min"})
        .reset_index()["timestamp_end"]
        .values
    )
    all_timestamps = df_str["timestamp_end"].values
    indices = np.where(np.in1d(earliest_timestamps, all_timestamps))[0]

    df_str.loc[:, ("earliest-occurence",)] = 0
    df_str.loc[indices, ("earliest-occurence",)] = 1
    df_str = df_str.sort_values("timestamp_end")
    df_str = df_str.reset_index(drop=True)

    if MODE == "max":
        select_better_than_baseline = df_str["objective"] >= baseline_perf
    else:
        select_better_than_baseline = df_str["objective"] <= baseline_perf

    df_str.loc[:, ("better-than-baseline",)] = 0
    df_str.loc[select_better_than_baseline, ("better-than-baseline",)] = 1
    df_str.loc[:, ("count",)] = (
        df_str["earliest-occurence"] & df_str["better-than-baseline"]
    ).astype(int)
    df_str.loc[:, ("count",)] = df_str["count"].cumsum()

    return df_str["timestamp_end"].to_numpy(), df_str["count"].to_numpy()


def plot_objective_multi(df, exp_config, output_dir, show):
    """Plot multiple objective curves with respect to time."""
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:

            exp_dfs = exp_df

            T = np.linspace(0, exp_config["t_max"], 50000)

            y_list = []
            for i, df_i in enumerate(exp_dfs):
                df_i = df_i.sort_values("timestamp_end")
                x, y = df_i.timestamp_end.to_numpy(), df_i.objective.cummin().to_numpy()
                f = interp1d(x, y, kind="previous", fill_value="extrapolate")
                y = f(T)
                y_list.append(y)

            y_list = np.asarray(y_list)
            y_mean = y_list.mean(axis=0)
            y_std = y_list.std(axis=0)
            y_se = y_std / np.sqrt(y_list.shape[0])

            plt.plot(
                T,
                y_mean,
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )
            plt.fill_between(
                T,
                y_mean - 1.96 * y_se,
                y_mean + 1.96 * y_se,
                facecolor=exp_config["data"][exp_name]["color"],
                alpha=0.3,
            )
            # plt.fill_between(T,
            #                  y_mean-1.96*y_std,
            #                  y_mean+1.96*y_std,
            #                  facecolor=exp_config["data"][exp_name]["color"],
            #                  alpha=0.3)
        else:
            exp_df = exp_df.sort_values("timestamp_end")
            x, y = exp_df.timestamp_end.to_numpy(), exp_df.objective.cummax().to_numpy()
            if "hartmann6D" in exp_name:
                y = y + 3.32237  # hartmann6D

            plt.plot(
                x,
                y,
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                marker=exp_config["data"][exp_name].get("marker", None),
                markevery=len(x) // 5,
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    if MODE == "min":
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")

    plt.ylabel(exp_config.get("ylabel", "Objective"))
    plt.xlabel("Search time (min.)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def process_for_test_objective(df, 
                               mode="max", 
                               max_budget=None):
    assert mode in ["min", "max"]
    
    if df.objective.dtype != np.float64:
        m = df.objective.str.startswith("F")
        df.loc[m,"objective"] = df.loc[m,"objective"].replace("F", "-1000000")
        df = df.astype({"objective": float})
        
    if mode == "min":
        df["objective"] = np.negative(df["objective"])
        if "m:objective_val" in df.columns:
            df["m:objective_val"] = np.negative(df["m:objective_val"])
        df["m:objective_test"] = np.negative(df["m:objective_test"])


    if mode == "max":
        df["objective_cummax"] = df["objective"].cummax()
    else:
        df["objective_cummax"] = df["objective"].cummin()
        
    df["m:budget_cumsum"] = df["m:budget"].cumsum()
    df["idx"] = df.index
    df = df.merge(df.groupby("objective_cummax")[["idx"]].first().reset_index(), on="objective_cummax")
    df.rename(columns={"idx_y": "max_idx"}, inplace=True)
    df.index  = df.idx_x.values
    del df["idx_x"]
    
    for idx in df["max_idx"]:
        if df.loc[idx, "m:budget"] < max_budget:
            df.loc[idx, "m:budget_cumsum"] = df.loc[idx, "m:budget_cumsum"] + max_budget
                
    return df


def aulc(x, y):
    """Compute the area under the learning curve (AULC)."""
    assert len(x) == len(y)
    a = 0
    # # x_min = 10 * 60 # average duration of random architecture
    x_min = 0
    x_max = x.max()
    for i in range(0,len(x)-1):
        if x[i] >= x_min:
            a += (x[i+1] - x[i])/(x_max-x_min) * y[i] 
    return round(a, 3)

def plot_test_objective_multi(df, exp_config, output_dir, show):
    """Plot multiple test objective curves with respect to time."""
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:

            exp_dfs = exp_df

            T = np.linspace(0, exp_config["t_max"], 50000)

            y_list = []
            for i, df_i in enumerate(exp_dfs):

                df_i = process_for_test_objective(
                    df_i.sort_values("timestamp_end"),
                    mode=MODE,
                    max_budget=exp_config["max_budget"],
                )
                x = df_i.loc[df_i["max_idx"]]["timestamp_end"].values
                y = df_i.loc[df_i["max_idx"]][exp_config["test_objective"]].values

                f = interp1d(x, y, kind="previous", fill_value="extrapolate")
                y = exp_config.get("best_objective", 1) - f(T)
                y_list.append(y)

            y_list = np.asarray(y_list)
            y_mean = y_list.mean(axis=0)
            y_std = y_list.std(axis=0)
            y_se = y_std / np.sqrt(y_list.shape[0])

            plt.plot(
                T,
                y_mean,
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )
            plt.fill_between(
                T,
                y_mean - 1.96 * y_se,
                y_mean + 1.96 * y_se,
                facecolor=exp_config["data"][exp_name]["color"],
                alpha=0.3,
            )

        else:

            exp_df = process_for_test_objective(
                exp_df.sort_values("timestamp_end"),
                mode=MODE,
                max_budget=exp_config["max_budget"],
                )
            x = exp_df.loc[exp_df["max_idx"]]["timestamp_end"].values
            y = exp_df.loc[exp_df["max_idx"]][exp_config["test_objective"]].values

            idx = np.unique(x, return_index=True, axis=0)[1]

            x = x[idx]
            y = y[idx]

            x = np.clip(np.concatenate([x, [exp_config["t_max"]]]), 0, exp_config["t_max"])
            y = np.clip(exp_config.get("best_objective", 1) - np.concatenate([y, [y[-1]]]), 0, 1)
            
            area = aulc(x, y)
            exp_config["data"][exp_name]["AULC"] = area
            
            plt.step(
                x[:],
                y[:],
                where="post",
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                marker=exp_config["data"][exp_name].get("marker", None),
                markevery=len(x) // 5,
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    # if MODE == "min":
    #     plt.legend(loc="upper right")
    # else:
    #     plt.legend(loc="lower right")
    plt.legend(loc=exp_config.get("legend", "best"))

    plt.ylabel("Test Regret")
    plt.xlabel("Search time (min.)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    plt.grid(which="minor", color="gray", linestyle=":")
    plt.grid(which="major", linestyle="-")
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def plot_objective_multi_iter(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            max_n_iters = 0
            y_list = []
            for i, exp_df in enumerate(exp_dfs):
                exp_df = exp_df.sort_values("timestamp_end")
                x, y = (
                    np.arange(len(exp_df)),
                    exp_df.objective.cummax().to_numpy(),
                )

                y = -y if MODE == "min" else y
                y_list.append(y.tolist())

                max_n_iters = max(max_n_iters, len(x))

            for i, y in enumerate(y_list):
                y = y + [y[-1]] * (max_n_iters - len(y))
                y_list[i] = y

            y_list = np.asarray(y_list)
            y_mean = y_list.mean(axis=0)
            y_stde = y_list.std(axis=0) / np.sqrt(len(y_list))

            plt_kwargs = dict(
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

            plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

            x = np.arange(max_n_iters)
            plt.plot(x, y_mean, **plt_kwargs)
            plt.fill_between(
                x,
                y_mean - y_stde,
                y_mean + y_stde,
                alpha=0.25,
                color=exp_config["data"][exp_name]["color"],
            )
        else:
            exp_df = exp_df.sort_values("timestamp_end")
            x, y = (
                np.arange(len(exp_df)),
                exp_df.objective.cummax().to_numpy(),
            )

            y = -y if MODE == "min" else y

            plt.step(
                x,
                y,
                where="post",
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel(exp_config.get("ylabel", "Objective"))
    plt.xlabel("#Evaluation")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, len(x) - 1)

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def plot_objective_multi_budget(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    default_budget = exp_config.get("default_budget", 1)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:

            exp_dfs = exp_df

            x_space = np.arange(1, 200 * default_budget)

            y_list = []
            x_max_i = 0
            for i, df_i in enumerate(exp_dfs):
                df_i = df_i.sort_values("timestamp_end")

                if "m:budget" not in df_i.columns:
                    df_i["m:budget"] = default_budget
                x = df_i["m:budget"].cumsum().to_numpy()
                x_max_i = max(x[-1], x_max_i)

                y = df_i.objective.cummax().to_numpy()
                f = interp1d(x, y, kind="previous", fill_value="extrapolate")
                y = f(x_space)

                y = -y if MODE == "min" else y
                y_list.append(y)

            y_list = np.asarray(y_list)
            y_list = y_list[:, :x_max_i]
            y_mean = y_list.mean(axis=0)
            y_stde = y_list.std(axis=0) / np.sqrt(len(y_list))

            plt_kwargs = dict(
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

            plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

            plt.plot(x_space[:x_max_i], y_mean, **plt_kwargs)
            plt.fill_between(
                x_space[:x_max_i],
                y_mean - y_stde,
                y_mean + y_stde,
                alpha=0.25,
                color=exp_config["data"][exp_name]["color"],
            )

        else:
            exp_df = exp_df.sort_values("timestamp_end")
            if "m:budget" not in exp_df.columns:
                exp_df["m:budget"] = default_budget
            x = exp_df["m:budget"].cumsum()

            y = exp_df.objective.cummax().to_numpy()

            y = -y if MODE == "min" else y

            plt.plot(
                x,
                y,
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel(exp_config.get("ylabel", "Objective"))
    plt.xlabel("Budget")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    # plt.xlim(0)

    if exp_config.get("xscale"):
        plt.xscale(exp_config.get("xscale"))
        # plt.xlim(10)

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()



def plot_objective_multi_duration(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df

            x_space = []

            for i, df_i in enumerate(exp_dfs):
                exp_dfs[i] = df_i.sort_values("timestamp_end")

                x = exp_dfs[i]["duration"].cumsum().to_numpy()
                x_space.append(x)

            x_space = np.sort(np.concatenate(x_space))

            y_list = []
            for i, df_i in enumerate(exp_dfs):

                x = df_i["duration"].cumsum().to_numpy()
                y = df_i.objective.cummax().to_numpy()

                x = np.concatenate([[0.001], x])
                y = np.concatenate([[0], y])

                f = interp1d(x, y, kind="previous", fill_value="extrapolate")
                y = f(x_space)

                y = -y if MODE == "min" else y
                y_list.append(y)

            y_list = np.asarray(y_list)
            y_mean = np.nanmean(y_list, axis=0)
            y_stde = np.nanstd(y_list, axis=0) / np.sqrt(len(y_list))

            plt_kwargs = dict(
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

            plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

            plt.plot(x_space, y_mean, **plt_kwargs)
            # plt.fill_between(
            #     x_space,
            #     y_mean - y_stde,
            #     y_mean + y_stde,
            #     alpha=0.25,
            #     color=exp_config["data"][exp_name]["color"],
            # )

        else:
            exp_df = exp_df.sort_values("timestamp_end")

            x = exp_df["duration"].cumsum().to_numpy()
            y = exp_df.objective.cummax().to_numpy()

            y = -y if MODE == "min" else y

            plt.plot(
                x,
                y,
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend()
    plt.ylabel(exp_config.get("ylabel", "Objective"))
    plt.xlabel("Time (hours)")

    if exp_config.get("ylim"):
        plt.ylim(*exp_config.get("ylim"))

    # plt.xlim(0)

    if exp_config.get("xscale"):
        plt.xscale(exp_config.get("xscale"))

    if exp_config.get("yscale"):
        plt.yscale(exp_config.get("yscale"))

    # ax = plt.gca()
    # ticker_freq = exp_config["t_max"] / 5
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    # ax.xaxis.set_major_formatter(minute_major_formatter)
    # ax.xaxis.set_major_formatter(hour_major_formatter)

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def compile_profile(df):
    history = []

    for _, row in df.iterrows():
        history.append((row["timestamp_start"], 1))
        history.append((row["timestamp_end"], -1))

    history = sorted(history, key=lambda v: v[0])
    nb_workers = 0
    timestamp = [0]
    n_jobs_running = [0]
    for time, incr in history:
        nb_workers += incr
        timestamp.append(time)
        n_jobs_running.append(nb_workers)

    return timestamp, n_jobs_running


def compute_num_workers(exp_name, exp_config):

    # priority to YAML config
    num_workers = exp_config["data"][exp_name].get("num_workers")
    if num_workers:
        return num_workers

    exp_name = exp_name.split("-")
    alg_name = exp_name[1]
    num_nodes = int(exp_name[6])
    num_ranks_per_node = int(exp_name[7])

    if alg_name in ["CBO", "HB"]:
        return num_nodes * num_ranks_per_node - 1
    else:
        return num_nodes * num_ranks_per_node


def plot_utilization_multi(df, exp_config, output_dir, show):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        num_workers = compute_num_workers(exp_name, exp_config)

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df
            T = np.linspace(0, exp_config["t_max"], 1000)
            y_list = []
            for i, exp_df in enumerate(exp_dfs):
                x, y = compile_profile(exp_df)
                f = interp1d(x, y, kind="previous", fill_value="extrapolate")
                y = f(T)
                y = np.asarray(y) / num_workers * 100
                y_list.append(y)

            y_list = np.asarray(y_list)
            y_mean = y_list.mean(axis=0)
            # y_std = y_list.std(axis=0)
            y_err = 1.96 * np.std(y_list, axis=0) / np.sqrt(len(exp_dfs))

            plt_kwargs = dict(
                color=exp_config["data"][exp_name]["color"],
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

            plt_kwargs["label"] = exp_config["data"][exp_name]["label"]

            plt.plot(T, y_mean, **plt_kwargs)
            plt.fill_between(
                T,
                y_mean - y_err,
                y_mean + y_err,
                alpha=0.25,
                color=exp_config["data"][exp_name]["color"],
            )

        else:
            x, y = compile_profile(exp_df)

            y = np.asarray(y) / num_workers * 100

            plt.step(
                x,
                y,
                where="post",
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                marker=exp_config["data"][exp_name].get("marker", None),
                markevery=len(x) // 5,
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend(loc="lower left")
    plt.ylabel("Effective Utilization (%)")
    plt.xlabel("Search time (min.)")

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    plt.ylim(0, 100)

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    if show:
        plt.show()
    plt.close()


def write_infos(df, exp_config, output_dir):
    output_file_name = f"infos.yaml"
    output_path = os.path.join(output_dir, output_file_name)

    infos = {}

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:
            exp_dfs = exp_df

            num_evaluations = []
            utilization = []
            best_objective = []
            best_objective_timestamp = []
            for i, exp_df in enumerate(exp_dfs):
                infos[exp_name] = {}
                if "t_max" in exp_config:
                    t_max = exp_config["t_max"]
                else:
                    t_max = exp_df.timestamp_end.max()

                exp_df = exp_df[exp_df.timestamp_end <= t_max]

                num_evaluations.append(len(exp_df))

                if i == 0:
                    num_workers = compute_num_workers(exp_name, exp_config)
                    infos[exp_name]["num_workers"] = num_workers

                # available compute time
                T_avail = t_max * num_workers
                T_eff = float(
                    (exp_df.timestamp_end - exp_df.timestamp_start).to_numpy().sum()
                )
                utilization.append(T_eff / T_avail)

                # compute best objective found
                idx_best = exp_df.objective.argmax()

                obj_best = exp_df.objective.iloc[idx_best]
                obj_best_timestamp = exp_df.timestamp_end.iloc[idx_best]
                best_objective.append(float(obj_best))
                best_objective_timestamp.append(float(obj_best_timestamp))

            data = num_evaluations
            loc = np.mean(data)
            err = 1.96 * np.std(data) / np.sqrt(len(exp_dfs))
            infos[exp_name]["num_evaluations"] = f"{loc} +/- {err}"

            data = utilization
            loc = np.mean(data)
            err = 1.96 * np.std(data) / np.sqrt(len(exp_dfs))
            infos[exp_name]["utilization"] = f"{loc} +/- {err}"

            data = best_objective
            loc = np.mean(data)
            err = 1.96 * np.std(data) / np.sqrt(len(exp_dfs))
            infos[exp_name]["best_objective"] = f"{loc} +/- {err}"

            data = best_objective_timestamp
            loc = np.mean(data)
            err = 1.96 * np.std(data) / np.sqrt(len(exp_dfs))
            infos[exp_name]["best_objective_timestamp"] = f"{loc} +/- {err}"

        else:
            infos[exp_name] = {}

            if "t_max" in exp_config:
                t_max = exp_config["t_max"]
            else:
                t_max = exp_df.timestamp_end.max()

            # Completed evaluations
            completed_cond = exp_df["timestamp_end"] <= t_max


            infos[exp_name]["num_evaluations"] = int(completed_cond.sum())

            num_workers = compute_num_workers(exp_name, exp_config)
            infos[exp_name]["num_workers"] = num_workers

            # available compute time
            T_avail = t_max * num_workers

            # Time consumed by workers with finished evaluations
            T_eff0 = float(exp_df.loc[completed_cond, "duration"].to_numpy().sum())

            # Time consumed by workers with unfinished evaluations
            incomplete_cond = (exp_df["timestamp_start"] <= t_max) & (exp_df["timestamp_end"] > t_max)
            start_times = exp_df.loc[incomplete_cond, "timestamp_start"].to_numpy()
            end_times = exp_df.loc[incomplete_cond, "timestamp_end"].to_numpy()
            end_times[:] = t_max
            T_eff1 = float((end_times - start_times).sum())
            infos[exp_name]["utilization"] = (T_eff0 + T_eff1) / T_avail
            
            exp_df = exp_df[completed_cond]

            # compute best objective found
            if MODE == "max":
                idx_best = exp_df.objective.argmax()
            else:
                idx_best = exp_df.objective.argmin()

            obj_best = exp_config.get("best_objective", 1) - exp_df.objective.iloc[idx_best]
            obj_best_timestamp = exp_df.timestamp_end.iloc[idx_best]
            
            infos[exp_name]["best_objective"] = float(obj_best)
            if "test_objective" in exp_config:
                obj_best_test = exp_config.get("best_objective", 1) - exp_df[exp_config["test_objective"]].iloc[idx_best]
                infos[exp_name]["best_objective_test"] = round(float(obj_best_test), 3)
            infos[exp_name]["best_objective_timestamp"] = float(obj_best_timestamp)

            infos[exp_name]["time_to_baseline"] = float(exp_df[exp_df["objective"] > exp_config["baseline_best"]].timestamp_end.min()/60)

            infos[exp_name]["aulc"] = round(float(exp_config["data"][exp_name]["AULC"]), 3)
            infos[exp_name]["mean_duration"] = round(float(exp_df.duration.mean()), 0)
            infos[exp_name]["std_duration"] = round(float(exp_df.duration.std()), 0)

    yaml_dump(output_path, infos)


def plot_count_better_than_best(df, exp_config, output_dir):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_dir, output_file_name)

    plt.figure()

    for exp_name, exp_df in df.items():

        if "rep" in exp_config["data"][exp_name]:

            exp_dfs = exp_df
            ...

        else:
            exp_df = exp_df.sort_values("timestamp_end")

            # x,  = exp_df.timestamp_end.to_numpy(), exp_df.objective.to_numpy()
            x, y = count_better(exp_df, exp_config["baseline_best"])

            plt.plot(
                x,
                y,
                label=exp_config["data"][exp_name]["label"],
                color=exp_config["data"][exp_name]["color"],
                marker=exp_config["data"][exp_name].get("marker", None),
                markevery=len(x) // 5,
                linestyle=exp_config["data"][exp_name].get("linestyle", "-"),
            )

    ax = plt.gca()
    ticker_freq = exp_config["t_max"] / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)

    if exp_config.get("title") and PRINT_TITLE:
        plt.title(exp_config.get("title"))

    plt.legend(loc="upper left")

    plt.ylabel("Models $>$ Baseline")
    plt.xlabel("Search time (min.)")

    if exp_config.get("xlim"):
        plt.xlim(*exp_config.get("xlim"))
    else:
        plt.xlim(0, exp_config["t_max"])

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=360)
    plt.close()


def generate_figures(config):
    global MODE, FILE_EXTENSION

    exp_root = config["data-root"]
    figures_dir = config.get("figures-root", "figures")
    show = config.get("show", False)
    MODE = config.get("mode", "max")
    FILE_EXTENSION = config.get("format", "png")

    plots = {
        # "scatter-iter": plot_scatter_multi_iter,
        # "scatter-budget": plot_scatter_multi_budget,
        "scatter-time": plot_scatter_multi,
        # "objective-iter": plot_objective_multi_iter,
        # "objective-time": plot_objective_multi,
        "test-objective-time": plot_test_objective_multi,
        # "objective-budget": plot_objective_multi_budget,
        # "objective-test-budget": plot_objective_test_multi_budget,
        # "objective-duration": plot_objective_multi_duration,
        "utilization": plot_utilization_multi,
    }
    plot_functions = {plots[k] for k in config.get("plots", plots.keys())}

    for exp_num, exp_config in config["experiments"].items():
        exp_dirname = str(exp_num)
        output_dir = os.path.join(figures_dir, exp_dirname)

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        df = load_results(exp_root, exp_config)

        for plot_func in plot_functions:
            plot_func(df, exp_config, output_dir, show)
        
        write_infos(df, exp_config, output_dir)

        if "baseline_best" in exp_config:
            plot_count_better_than_best(df, exp_config, output_dir)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line to plot experiments results."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="plot.yaml",
        help="Plotter configuration YAML file.",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    yaml_path = args.config
    config = yaml_load(yaml_path)
    generate_figures(config)
    print("Done!")

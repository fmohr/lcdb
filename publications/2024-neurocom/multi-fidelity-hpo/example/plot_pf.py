import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deephyper.analysis._matplotlib import figure_size, update_matplotlib_rc


update_matplotlib_rc()
figsize = figure_size(252 * 1.0, 1.0)


class Experiment:
    problem = "dhexp.benchmark.hpobench_tabular"
    search = "RANDOM"
    stopper = None
    stopper_args = None
    max_evals = 200
    max_budget = 100
    random_state = None

    def __init__(self, stopper, stopper_args, random_state) -> None:
        self.stopper = stopper
        self.stopper_args = stopper_args
        self.random_state = random_state

    @property
    def log_dir(self):
        return f"output/{self.problem}-{self.search}-{self.stopper}-{self.stopper_args}-{self.max_evals}-{self.random_state}"

    @property
    def path_results(self):
        return os.path.join(self.log_dir, "results.csv")

    def load_results(self):
        return pd.read_csv(self.path_results, index_col=None)


import itertools

random_states = "1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113".split()


# Pareto-Front
map_stopper_to_args = {
    "deephyper.stopper.ConstantStopper": "1 5 100".split(),
    "deephyper.stopper.SuccessiveHalvingStopper": "64 4 1.19".split(),
    # "deephyper.stopper.LCModelStopper.pow3": "0.5 0.7 0.8 0.9 0.95".split(),
    # "deephyper.stopper.LCModelStopper.mmf4": "0.5 0.7 0.8 0.9 0.95".split(),
    # "deephyper_benchmark.stopper.lcpfn.LCPFNStopper": "0.5 0.7 0.8 0.9 0.95".split(),
}

topk_ = 3
map_stopper_to_topk = {k: topk_ for k in map_stopper_to_args.keys()}

linestyle_list = {
    i: v
    for i, v in enumerate(
        [
            "-",
            "--",
            ":",
        ]
    )
}

map_stopper_to_letter = {
    "deephyper.stopper.ConstantStopper": "$i$",
    "deephyper.stopper.SuccessiveHalvingStopper": "$r$",
    "deephyper.stopper.LCModelStopper.pow3": "$\\rho$",
    "deephyper.stopper.LCModelStopper.mmf4": "$\\rho$",
    "deephyper_benchmark.stopper.lcpfn.LCPFNStopper": "$\\rho$",
}

map_stopper_to_color = {
    "deephyper.stopper.ConstantStopper": "C3",
    "deephyper.stopper.SuccessiveHalvingStopper": "C1",
    "deephyper_benchmark.stopper.lcpfn.LCPFNStopper": "C0",
    "deephyper.stopper.LCModelStopper.pow3": "C4",
    "deephyper.stopper.LCModelStopper.mmf4": "C2",
}

map_stopper_to_label = {
    "deephyper.stopper.ConstantStopper": "{}-Epoch",
    "deephyper.stopper.SuccessiveHalvingStopper": "{}-SHA",
    "deephyper_benchmark.stopper.lcpfn.LCPFNStopper": "{}-PFN",
    "deephyper.stopper.LCModelStopper.pow3": "{}-POW3",
    "deephyper.stopper.LCModelStopper.mmf4": "{}-LCE",
}

data = {
    "deephyper.stopper.SuccessiveHalvingStopper": {},
    # "deephyper_benchmark.stopper.lcpfn.LCPFNStopper": {},
    # "deephyper.stopper.LCModelStopper.pow3": {},
    # "deephyper.stopper.LCModelStopper.mmf4": {},
    "deephyper.stopper.ConstantStopper": {},
}


for stopper, stopper_items in data.items():
    for stopper_args, random_state in itertools.product(
        map_stopper_to_args[stopper], random_states
    ):
        exp = Experiment(stopper, stopper_args, random_state)
        df = exp.load_results()
        dfs = stopper_items.get(stopper_args, [])
        dfs.append(df)
        stopper_items[stopper_args] = dfs


def process(
    df,
    mode="max",
    topk_tournament=None,
    filter_duplicates=False,
    max_budget=100,
    consider_checkpointed_weights=False,
):
    assert mode in ["min", "max"]

    if df.objective.dtype != np.float64:
        m = df.objective.str.startswith("F")
        df.loc[m, "objective"] = df.loc[m, "objective"].replace("F", "-1000000")
        df = df.astype({"objective": float})

    if mode == "min":
        df["objective"] = np.negative(df["objective"])
        if "m:objective_val" in df.columns:
            df["m:objective_val"] = np.negative(df["m:objective_val"])
        df["m:objective_test"] = np.negative(df["m:objective_test"])

    if topk_tournament:
        k = topk_tournament
        max_idx = []
        df["m:budget_cumsum"] = df["m:budget"].cumsum()
        for i in range(len(df)):
            fdf = df[: i + 1]

            if filter_duplicates:
                fdf = fdf.drop_duplicates(
                    [pname for pname in df.columns if "p:" in pname], keep="last"
                )

            if mode == "max":
                topk = fdf[: i + 1].nlargest(n=k, columns="objective")
                if k == 1:
                    winner = topk
                else:
                    winner = topk.nlargest(n=1, columns="m:objective_val")
            else:
                topk = fdf[: i + 1].nsmallest(n=k, columns="objective")
                if k == 1:
                    winner = topk
                else:
                    winner = topk.nsmallest(n=1, columns="m:objective_val")

            # consider that checkpointed "weights" can be reloaded
            if consider_checkpointed_weights:
                df.loc[i, "m:budget_cumsum"] = (
                    df.loc[i, "m:budget_cumsum"]
                    + len(topk) * 100
                    - topk["m:budget"].sum()
                )

            # consider that selected models are retrained from scratch
            else:
                for topk_i in range(min(len(topk), k)):
                    budget = topk.iloc[topk_i]["m:budget"]
                    # retraining cost if not completed
                    if budget < max_budget:
                        df.loc[i, "m:budget_cumsum"] = (
                            df.loc[i, "m:budget_cumsum"] + max_budget
                        )

            winner_idx = winner.index.tolist()[0]
            max_idx.append(winner_idx)

        df["max_idx"] = max_idx

    else:
        if mode == "max":
            df["objective_cummax"] = df["objective"].cummax()
        else:
            df["objective_cummax"] = df["objective"].cummin()

        df["m:budget_cumsum"] = df["m:budget"].cumsum()
        df["idx"] = df.index
        df = df.merge(
            df.groupby("objective_cummax")[["idx"]].first().reset_index(),
            on="objective_cummax",
        )
        df.rename(columns={"idx_y": "max_idx"}, inplace=True)
        df.index = df.idx_x.values
        del df["idx_x"]

        for idx in df["max_idx"]:
            if df.loc[idx, "m:budget"] < max_budget:
                df.loc[idx, "m:budget_cumsum"] = (
                    df.loc[idx, "m:budget_cumsum"] + max_budget
                )

    return df


max_evals = exp.max_evals
max_budget = exp.max_budget
plot_val = False
plot_regret = True

# https://matplotlib.org/stable/tutorials/colors/colormaps.html#lightness-of-matplotlib-colormaps
cmap = mpl.colormaps["tab10"]

data_y_final = {k: [] for k in data.keys()}

pf_x = []
pf_y = []

y_max_all = -np.inf
y_min_all = np.inf

for i, (stopper, stopper_items) in enumerate(data.items()):
    for j, (stopper_args, dfs) in enumerate(stopper_items.items()):
        print(stopper, stopper_args)
        color = map_stopper_to_color.get(stopper, cmap(i))
        linestyle = linestyle_list.get(j, "-")

        x_space = np.arange(max_budget, max_evals * max_budget + 1)
        dfs_processed = map(
            lambda x: process(
                x,
                mode="max",
                topk_tournament=map_stopper_to_topk[stopper],
                max_budget=max_budget,
            ),
            dfs,
        )
        x_list = []
        y_list = []
        for j, df in enumerate(dfs_processed):
            x = df["m:budget_cumsum"].to_numpy()

            if plot_val:
                y = df.loc[df["max_idx"]]["m:objective_val"].values
            else:
                y = df.loc[df["max_idx"]]["m:objective_test"].values

            x_list.append(x)
            y_list.append(y)

        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)

        x_mean = np.mean(x_list, axis=0)
        x_stde = np.std(x_list, axis=0) / np.sqrt(x_list.shape[0])

        # regret
        if plot_regret:
            y_list = 1 - y_list
        y_mean = np.mean(y_list, axis=0)

        y_stde = np.std(y_list, axis=0) / np.sqrt(y_list.shape[0])
        y_min = y_mean - y_stde
        y_max = y_mean + y_stde

        mask = x_mean >= max_budget * 3
        y_max_all = max(y_max_all, np.max(y_max[mask]))
        y_min_all = min(y_min_all, np.min(y_min[mask]))

        data_y_final[stopper].append([x_mean[-1], y_mean[-1], x_stde[-1], y_stde[-1]])

from deephyper.skopt.moo import pareto_front, hypervolume

output_info = ""
y_all = []
y_all_stde = []

for i, (stopper, y) in enumerate(data_y_final.items()):
    y = np.array(y)
    y_all.extend(y[:, :2])
    y_all_stde.extend(y[:, 2:])
y_all = np.array(y_all)
y_all_stde = np.array(y_all_stde)

pf_all = pareto_front(y_all, sort=True)
ref = np.max(y_all + y_all_stde, axis=0)

hv_all = hypervolume(np.log10(y_all / ref), ref=np.log10([1, 1]))
output_info += f"ALL - HVI={hv_all:.3f}\n"

plt.figure(figsize=figsize)
for i, (stopper, y) in enumerate(data_y_final.items()):
    y = np.array(y)
    pf, pf_idx = pareto_front(y[:, :2], sort=True, return_idx=True)

    hv = hypervolume(np.log10(pf / ref), ref=np.log10([1, 1]))
    ratio = hv / hv_all

    label = map_stopper_to_label[stopper].format(map_stopper_to_letter[stopper])
    output_info += f"{label} - HVI={hv:.3f} - HVI/HVI_ALL={ratio:.3f}\n"

    plt.scatter(y[:, 0], y[:, 1], marker="o", color=map_stopper_to_color[stopper], s=10)

    # if "ConstantStopper" in stopper:
    plt.step(
        pf[:, 0],
        pf[:, 1],
        color=map_stopper_to_color[stopper],
        linestyle="-",
        where="post",
        label=map_stopper_to_label[stopper].format(map_stopper_to_letter[stopper]),
    )

    # plt.step(
    #     pf[:, 0] + y[pf_idx, 2:3].reshape(-1),
    #     pf[:, 1] + y[pf_idx, 3:4].reshape(-1),
    #     color=map_stopper_to_color[stopper],
    #     linestyle="--",
    #     where="post",
    # )

    # plt.step(
    #     pf[:, 0] - y[pf_idx, 2:3].reshape(-1),
    #     pf[:, 1] - y[pf_idx, 3:4].reshape(-1),
    #     color=map_stopper_to_color[stopper],
    #     linestyle="--",
    #     where="post",
    # )

    y0 = pf[:, 0].tolist()
    y1 = pf[:, 1].tolist()
    # make y0,y1 a step function
    y0_, y1_ = [], []
    for i in range(len(y0) - 1):
        y0_.extend([y0[i], y0[i + 1]])
        y1_.extend([y1[i], y1[i]])
    y0_.append(y0[-1])
    y1_.append(y1[-1])
    y0 = y0_
    y1 = y1_
    y0 = [y0[0]] + y0 + [ref[0], ref[0]]
    y1 = [ref[1]] + y1 + [y1[-1], ref[1]]
    plt.fill(y0, y1, alpha=0.25, color=map_stopper_to_color[stopper])

    # plt.errorbar(
    #     y[:, 0],
    #     y[:, 1],
    #     xerr=y[:, 2:3].reshape(-1),
    #     yerr=y[:, 3:4].reshape(-1),
    #     color=map_stopper_to_color[stopper],
    #     linestyle="none",
    #     marker="none",
    # )

plt.step(
    pf_all[:, 0],
    pf_all[:, 1],
    color="black",
    linewidth=1,
    linestyle=":",
    where="post",
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Training Epochs Consumed")
if plot_val:
    plt.ylabel("$1 -$ Validation $R^2$")
else:
    plt.ylabel("$1 -$ Test $R^2$")
lgnd = plt.legend(ncol=4, loc="upper right", bbox_to_anchor=(1.0, 1.2))
for i in range(len(lgnd.legend_handles)):
    lgnd.legend_handles[i]._sizes = [40]
plt.grid()
plt.grid(visible=True, which="minor", color="gray", linestyle=":")
plt.savefig(os.path.join("figures", "pf.jpg"), dpi=300, bbox_inches="tight")
# plt.show()

with open(os.path.join("figures", "pf.txt"), "w") as f:
    f.write(output_info)

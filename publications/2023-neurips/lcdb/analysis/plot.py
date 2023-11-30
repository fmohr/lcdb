import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from deephyper.analysis import rank


def plot_learning_curves(fidelity_values, metric_values, mode="max", rank_method="dense", decimals=5, alpha=1.0, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    cmap = mpl.colormaps["seismic"]

    # corresponds to iloc indexing
    metric_max_fidelity = np.asarray([y[-1] for y in metric_values])

    if mode == "max":
        ranking = rank(-metric_max_fidelity, decimals=decimals, method=rank_method)
    elif mode == "min":
        ranking = rank(metric_max_fidelity, decimals=decimals, method=rank_method)
    else:
        raise ValueError(f"Unknown mode '{mode}' should be 'max' or 'min'.")
    
    ranking_max = ranking.max()
    for i, (x, y) in enumerate(zip(fidelity_values, metric_values)):
        ax.plot(x, y, color=cmap(ranking[i] / ranking_max), alpha=alpha)

    ax.grid()

    norm = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
    norm.set_clim(1, ranking_max)
    plt.colorbar(norm, ax=plt.gca(), label="Rank")
    # plt.xlim(0, fidelities.max())

    return fig, ax

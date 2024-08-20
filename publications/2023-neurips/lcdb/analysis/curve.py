from ..db import LCDB
from .score import balanced_accuracy_from_confusion_matrix
from .json import QueryMetricValuesFromAnchors, QueryAnchorValues

import numpy as np


def get_sample_wise_curves(df=None, workflows=None, datasets=None, metric="error_rate", lcdb=None, rounding_decimals=4):
    """
    Computes the sample-wise learning curve for a specific metric for a set of configurations, possibly across workflows and datasets.

    :param df: A dataframe with the raw data retrieved from an LCDB object via `get_results`. If none is given, this is computed automatically. `workflows`, `datasets`, and `lcdb` are only used if this field is `None`
    :param workflows: iterable of workflows for which curves should be computed. If `None`, curves are computed for all available workflows.
    :param datasets: iterable of datasets for which curves should be computed. If `None`, curves are computed for all available datasets.
    :param metric: the metric for which the learning curve should be computed. Must be one of the following: `error_rate` (default)
    :param lcdb: An object for the learning curve database from which results are retrieved. If `None`, the standard object is used.
    :param rounding_decimals: number of decimals to which the values should be rounded.
    :return: Pandas dataframe with a column named `sw_<metric>`, where `metric` is the given metric.
    """

    # check whether requested metric is valid
    accepted_metrics = ["error_rate"]
    if metric == "error_rate":
        fun = lambda cm: 1 - balanced_accuracy_from_confusion_matrix(cm)
        src = "confusion_matrix"
    else:
        raise ValueError(f"metric is {metric} but must be in {accepted_metrics}.")

    # get result dataframe (if not provided)
    if df is None:
        if lcdb is None:
            lcdb = LCDB()
        df = lcdb.get_results(workflows=workflows, openmlids=datasets)

    # compute
    s_anchors = df["m:json"].apply(QueryAnchorValues()).to_list()
    s_values = df["m:json"].apply(
        lambda row: [
            np.round(fun(entry), decimals=rounding_decimals)
            for entry in QueryMetricValuesFromAnchors(src)(row)
        ]
    )
    df[f"sw_{metric}"] = [list(zip(a, v)) for a, v in zip(s_anchors, s_values)]
    return df

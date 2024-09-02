import pandas as pd

from .score import balanced_accuracy_from_confusion_matrix
from .json import QueryMetricValuesFromAnchors, QueryAnchorValues, QueryEpochValues, QueryMetricValuesFromEpochs

import numpy as np
import itertools as it


class LearningCurve:

    def __init__(self, hp_config, openmlid, values, metrics, val_seeds, test_seeds, wf_seeds, anchors_size, anchors_iteration=None):
        self.hp_config = hp_config
        self.openmlid = openmlid
        self.values = values
        self.anchors_size = anchors_size
        self.anchors_iteration = anchors_iteration

class LearningCurveExtractor:

    def __init__(self, metrics=["error_rate"], folds=["train", "val", "test"], rounding_decimals=4, return_none_on_error=True):
        accepted_metrics = ["error_rate", "balanced_error_rate"]

        self.funs = {}
        self.srcs = {}
        self.folds = folds

        for metric in metrics:
            if metric == "error_rate":
                self.funs[metric] = lambda cm: 1 - np.diag(cm).sum() / np.sum(cm)
                self.srcs[metric] = "confusion_matrix"
            elif metric == "balanced_error_rate":
                self.funs[metric] = lambda cm: 1 - balanced_accuracy_from_confusion_matrix(cm)
                self.srcs[metric] = "confusion_matrix"
            else:
                raise ValueError(f"metric is {metric} but must be in {accepted_metrics}.")

        self.metrics = metrics
        self.rounding_decimals = rounding_decimals
        self.return_none_on_error = return_none_on_error

    def __call__(self, lc_dict):
        """
        Computes the sample-wise learning curve for a specific metric for a set of configurations, possibly across workflows and datasets.
        """

        # determine anchors and whether this is an iteration curve
        try:
            anchors_size = QueryAnchorValues()(lc_dict)
            anchors_iterations_per_sample_size = QueryEpochValues()(lc_dict)
            anchors_iteration = set()
            for iteration_anchors in anchors_iterations_per_sample_size:
                anchors_iteration |= set(iteration_anchors)
            anchors_iteration = sorted(anchors_iteration)
            is_iteration_curve = len(anchors_iteration) > 0

            # create array for values of curve
            shape = (len(self.metrics), len(self.folds), len(anchors_size), len(anchors_iteration)) if is_iteration_curve else (len(self.metrics), len(self.folds), len(anchors_size))
            values = np.zeros(shape)
            values[:] = np.nan

            for i1, metric in enumerate(self.metrics):
                fun = self.funs[metric]
                for i2, fold in enumerate(self.folds):

                    if is_iteration_curve:
                        sources_for_iteration_curve_values = QueryMetricValuesFromEpochs(self.srcs[metric], split_name=fold)(lc_dict)
                        for i3, (anchor_size, sources_for_anchor_size) in enumerate(zip(anchors_size, sources_for_iteration_curve_values)):
                            for anchor_iteration, source_for_lc_point in zip(anchors_iterations_per_sample_size[i3], sources_for_anchor_size):
                                i4 = anchors_iteration.index(anchor_iteration)
                                values[i1, i2, i3, i4] = fun(source_for_lc_point)
                    else:
                        values[i1, i2] = [fun(e) for e in QueryMetricValuesFromAnchors(self.srcs[metric], split_name=fold)(lc_dict)]

            lc_params = {
                "hp_config": None,
                "openmlid": None,
                "values": values,
                "metrics": self.metrics,
                "val_seeds": None,
                "test_seeds": None,
                "wf_seeds": None,
                "anchors_size": anchors_size
            }
            if is_iteration_curve:
                lc_params["anchors_iteration"] = anchors_iteration
            return LearningCurve(**lc_params)

        except KeyboardInterrupt:
            raise

        except:
            raise


class IterationWiseCurveExtractor:

    def __init__(self, metrics=["error_rate"], folds=["train", "val", "test"], rounding_decimals=4):
        accepted_metrics = ["error_rate", "balanced_error_rate"]

        self.funs = {}
        self.srcs = {}
        self.folds = folds

        for metric in metrics:
            if metric == "error_rate":
                self.funs[metric] = lambda cm: 1 - np.diag(cm).sum() / np.sum(cm)
                self.srcs[metric] = "confusion_matrix"
            elif metric == "balanced_error_rate":
                self.funs[metric] = lambda cm: 1 - balanced_accuracy_from_confusion_matrix(cm)
                self.srcs[metric] = "confusion_matrix"
            else:
                raise ValueError(f"metric is {metric} but must be in {accepted_metrics}.")

        self.metrics = metrics
        self.rounding_decimals = rounding_decimals

    def __call__(self, lc_dict):
        """
        Computes the iteration-wise learning curve for each sample-wise anchor and a specific metric for a set of configurations, possibly across workflows and datasets.
        """

        anchors_sizes = QueryAnchorValues()(lc_dict)
        anchors_iterations_per_sample_size = QueryEpochValues()(lc_dict)

        data = {}

        for metric in self.metrics:
            for fold in self.folds:
                key = f"{metric}_{fold}"
                values = QueryMetricValuesFromEpochs(self.srcs[metric], split_name=fold)(lc_dict)

                data[key] = []

                anchor_size_column = []
                anchor_iter_column = []
                for i, (anchor_size, anchors_iteration_for_this_sample_anchor) in enumerate(zip(anchors_sizes, anchors_iterations_per_sample_size)):
                    num_values_for_iteration_curve = len(values[i])
                    for j, anchor_iteration in enumerate(anchors_iteration_for_this_sample_anchor):
                        if j >= num_values_for_iteration_curve:
                            break
                        data[key] = self.funs[metric](values[i][j])
                        if "anchor_size" not in data:
                            anchor_size_column.append(anchor_size)
                            anchor_iter_column.append(anchor_iteration)
                if "anchor_size" not in data:
                    data["anchor_size"] = anchor_size_column
                    data["anchor_iteration"] = anchor_iter_column
        return pd.DataFrame(data)


def integrate_sample_wise_curves(df_results, column_with_sample_wise_curves):
    dfs = []
    if column_with_sample_wise_curves not in df_results.columns:
        raise ValueError(f"Cannot extract sample-wise curves from inexistent field {column_with_sample_wise_curves}")
    cols = list(df_results.columns)
    cols.remove(column_with_sample_wise_curves)
    for _, row in df_results.iterrows():
        df = row[column_with_sample_wise_curves]
        sub_cols = list(df.columns)
        for key in cols:
            df[key] = row[key]
        df = df[cols + sub_cols]
        dfs.append(df)
    return pd.concat(dfs)

def to_numpy(df_results, curve_column):

    hp_cols = [c for c in df_results.columns if c.startswith("p:")]
    num_configs = len(df_results.groupby(hp_cols))
    num_val_seeds = len(pd.unique(df_results["m:valid_seed"]))
    num_test_seeds = len(pd.unique(df_results["m:test_seed"]))
    num_wf_seeds = len(pd.unique(df_results["m:workflow_seed"]))

    tensor = np.zeros((num_configs, num_val_seeds, num_test_seeds, num_wf_seeds, df_results[curve_column].iloc[0].shape[1] - 1, df_results[curve_column].iloc[0].shape[0]))
    tensor[:] = np.nan

    anchors_sizes_overall = None

    for i1, (hp_conf, df_i1) in enumerate(df_results.groupby(hp_cols)):
        for i2, (val_seed, df_i2) in enumerate(df_i1.groupby("m:valid_seed")):
            for i3, (test_seed, df_i3) in enumerate(df_i2.groupby("m:test_seed")):
                for i4, (wf_seed, df_i4) in enumerate(df_i3.groupby("m:workflow_seed")):
                    lc = df_i4[curve_column].iloc[0]

                    is_iteration_curve = "anchor_iteration" in lc.columns

                    anchors_size = np.array(sorted(set(lc["anchor_size"])))
                    if is_iteration_curve:
                        anchors_iteration = np.array(sorted(set(lc["anchor_iteration"])))
                        lc = lc.set_index(["anchor_size", "anchor_iteration"])
                    else:
                        lc = lc.set_index(["anchor_size"])

                    if anchors_sizes_overall is None:
                        anchors_sizes_overall = anchors_size
                    else:
                        if np.any(anchors_sizes_overall != anchors_size):
                            raise ValueError()

                    print(lc)
                    print(lc.values)

                    # if this is a sample-wise curve, check anchors
                    tensor[i1, i2, i3, i4, :, :lc.shape[0]] = lc.values.T
    return tensor

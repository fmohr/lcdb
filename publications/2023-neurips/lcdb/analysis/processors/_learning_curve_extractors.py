import numpy as np
import pandas as pd

from lcdb.analysis.json import (
    QueryAnchorValues,
    QueryEpochValues,
    QueryMetricValuesFromAnchors,
    QueryMetricValuesFromEpochs,
)
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix
from lcdb.analysis._learning_curves import LearningCurve

DETAIL_KEY = "results"


class LearningCurveExtractor:

    def __init__(
        self,
        metrics=["error_rate"],
        folds=["train", "val", "test"],
        rounding_decimals=4,
        return_none_on_error=True,
    ):
        accepted_metrics = ["error_rate", "balanced_error_rate"]

        self.funs = {}
        self.srcs = {}
        self.folds = folds

        for metric in metrics:
            if not isinstance(metric, str):
                raise ValueError(f"Each metric in metrics must be str, but at least one is {type(metric)}: {metric}")

            if metric == "error_rate":
                self.funs[metric] = lambda cm: 1 - np.diag(cm).sum() / np.sum(cm)
                self.srcs[metric] = "confusion_matrix"
            elif metric == "balanced_error_rate":
                self.funs[metric] = (
                    lambda cm: 1 - balanced_accuracy_from_confusion_matrix(cm)
                )
                self.srcs[metric] = "confusion_matrix"
            else:
                raise ValueError(
                    f"metric is {metric} but must be in {accepted_metrics}."
                )

        self.metrics = metrics
        self.rounding_decimals = rounding_decimals
        self.return_none_on_error = return_none_on_error

    def __call__(self, row):
        """
            Computes the sample-wise learning curve for a specific metric for a set of configurations, possibly across workflows and datasets.
        """
        if DETAIL_KEY not in row or row[DETAIL_KEY] is None or len(row[DETAIL_KEY]) == 0:
            return {"learning_curve": None}

        lc_dict = row[DETAIL_KEY]

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
            shape = (
                (
                    len(self.metrics),
                    len(self.folds),
                    1,
                    1,
                    1,
                    len(anchors_size),
                    len(anchors_iteration),
                )
                if is_iteration_curve
                else (len(self.metrics), len(self.folds), 1, 1, 1, len(anchors_size))
            )
            values = np.zeros(shape)
            values[:] = np.nan

            for i1, metric in enumerate(self.metrics):
                fun = self.funs[metric]
                for i2, fold in enumerate(self.folds):

                    if is_iteration_curve:
                        sources_for_iteration_curve_values = (
                            QueryMetricValuesFromEpochs(
                                self.srcs[metric], split_name=fold
                            )(lc_dict)
                        )
                        for i3, (anchor_size, sources_for_anchor_size) in enumerate(
                            zip(anchors_size, sources_for_iteration_curve_values)
                        ):
                            for anchor_iteration, source_for_lc_point in zip(
                                anchors_iterations_per_sample_size[i3],
                                sources_for_anchor_size,
                            ):
                                i4 = anchors_iteration.index(anchor_iteration)
                                values[i1, i2, 0, 0, 0, i3, i4] = fun(source_for_lc_point)
                    else:
                        sample_wise_curve = [
                            fun(e)
                            for e in QueryMetricValuesFromAnchors(
                                self.srcs[metric], split_name=fold
                            )(lc_dict)
                        ]
                        num_missing_entries = values.shape[-1] - len(sample_wise_curve)
                        if num_missing_entries > 0:
                            sample_wise_curve.extend(num_missing_entries * [np.nan])
                        values[i1, i2, 0, 0, 0] = sample_wise_curve

            lc_params = {
                "workflow": row["workflow"],
                "hp_config": row["config"],
                "openmlid": row["openmlid"],
                "values": values,
                "metrics": self.metrics,
                "fold_names": self.folds,
                "test_seeds": [row["test_seed"]],
                "val_seeds": [row["valid_seed"]],
                "workflow_seeds": [row["workflow_seed"]],
                "anchors_size": anchors_size,
            }
            if is_iteration_curve:
                lc_params["anchors_iteration"] = anchors_iteration
            return {"learning_curve": LearningCurve(**lc_params)}

        except KeyboardInterrupt:
            raise

        except:
            raise


class IterationWiseCurveExtractor:

    def __init__(
        self,
        metrics=["error_rate"],
        folds=["train", "val", "test"],
        rounding_decimals=4,
    ):
        accepted_metrics = ["error_rate", "balanced_error_rate"]

        self.funs = {}
        self.srcs = {}
        self.folds = folds

        for metric in metrics:
            if metric == "error_rate":
                self.funs[metric] = lambda cm: 1 - np.diag(cm).sum() / np.sum(cm)
                self.srcs[metric] = "confusion_matrix"
            elif metric == "balanced_error_rate":
                self.funs[metric] = (
                    lambda cm: 1 - balanced_accuracy_from_confusion_matrix(cm)
                )
                self.srcs[metric] = "confusion_matrix"
            else:
                raise ValueError(
                    f"metric is {metric} but must be in {accepted_metrics}."
                )

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
                values = QueryMetricValuesFromEpochs(
                    self.srcs[metric], split_name=fold
                )(lc_dict)

                data[key] = []

                anchor_size_column = []
                anchor_iter_column = []
                for i, (
                    anchor_size,
                    anchors_iteration_for_this_sample_anchor,
                ) in enumerate(zip(anchors_sizes, anchors_iterations_per_sample_size)):
                    num_values_for_iteration_curve = len(values[i])
                    for j, anchor_iteration in enumerate(
                        anchors_iteration_for_this_sample_anchor
                    ):
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

import itertools as it
from collections import OrderedDict

import numpy as np
import pandas as pd
from lcdb.timer import Timer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


class Curve:
    """
    Entity class to compute and store information about performance (and time to compute the metrics)
    """

    def __init__(self, workflow=None, timer=None):
        self.workflow = workflow
        self.timer = timer if timer is not None else Timer()
        self.curve_data = OrderedDict()

    def __len__(self):
        """The number of anchors in the curve corresponds to the length of the curve."""
        return len(self.curve_data)

    def __getitem__(self, anchor):
        """Get the data for a given anchor."""
        return self.curve_data[anchor]

    def compute_metrics(self, anchor, y_true, y_pred, y_pred_proba):
        if self.workflow is None:
            raise ValueError("This is a read-only curve. No workflow is given.")
        relevant_labels = self.workflow.infos["classes"].copy()
        if not isinstance(relevant_labels, list):
            raise ValueError(
                f"infos['classes'] must be a list but is {type(relevant_labels)}"
            )
        labels_in_true_data_not_used_by_workflow = list(
            set(y_true).difference(relevant_labels)
        )
        if len(labels_in_true_data_not_used_by_workflow) > 0:
            expansion_matrix = np.zeros(
                (len(y_true), len(labels_in_true_data_not_used_by_workflow))
            )
            relevant_labels.extend(labels_in_true_data_not_used_by_workflow)
            y_pred_proba = np.column_stack([y_pred_proba, expansion_matrix])
            y_pred_proba = np.column_stack(
                [y_pred_proba[:, i] for i in np.argsort(relevant_labels)]
            )
            relevant_labels = sorted(relevant_labels)

        is_binary = len(np.unique(y_true)) == 2

        # sanity check: should not compute an anchor several times
        if anchor in self.curve_data:
            raise ValueError(f"Data for anchor {anchor} already available")
        self.curve_data[anchor] = {}

        for target in ["cm", "accuracy", "auc", "log_loss", "brier_score"]:
            self.timer.start(f"metric_{target}")
            if target == "cm":
                score = np.round(
                    confusion_matrix(y_true, y_pred, labels=relevant_labels), 5
                )

            elif target == "accuracy":
                # TODO: why not balanced accuracy?
                score = np.round(accuracy_score(y_true, y_pred), 5)
            elif target == "auc":
                if is_binary:
                    score = np.round(
                        roc_auc_score(
                            y_true, y_pred_proba[:, 1], labels=relevant_labels
                        ),
                        5,
                    )
                else:
                    score = {}
                    for multi_class, average in it.product(
                        ["ovr", "ovo"], ["micro", "macro", "weighted", None]
                    ):
                        if average in [None, "micro"] and multi_class != "ovr":
                            continue
                        auc = np.round(
                            roc_auc_score(
                                y_true,
                                y_pred_proba,
                                labels=relevant_labels,
                                multi_class=multi_class,
                                average=average,
                            ),
                            5,
                        )
                        score[f"auc_{multi_class}_{average}"] = auc
            elif target == "log_loss":
                y_base = y_pred_proba[:, 1] if is_binary else y_pred_proba
                score = np.round(log_loss(y_true, y_base, labels=relevant_labels), 5)
            elif target == "brier":
                if is_binary:
                    score = np.round(
                        brier_score_loss(
                            y_true, y_pred_proba[:, 1], pos_label=relevant_labels[1]
                        ),
                        5,
                    )
                else:
                    y_true_binarized = np.zeros((len(y_true), len(relevant_labels)))
                    for j, label in enumerate(relevant_labels):
                        mask = y_true == label
                        y_true_binarized[mask, j] = 1
                    score = np.round(
                        ((y_true_binarized - y_pred_proba) ** 2).sum(axis=1).mean(), 5
                    )

            # store results and time
            self.timer.stop(f"metric_{target}")
            if type(score) == dict:
                self.curve_data[anchor].update(score)
            else:
                self.curve_data[anchor][f"{target}"] = score

    @property
    def anchors(self):
        # return sorted(self.curve_data.keys())
        return list(self.curve_data.keys())

    def as_dataframe(self):
        """
        Formats the curve knowledge as a dataframe, with one row per anchor

        :return: DataFrame
        """
        anchors = self.anchors
        keys = list(self.curve_data[anchors[0]].keys())
        rows = []
        for anchor in anchors:
            rows.append([anchor] + [self.curve_data[anchor][key] for key in keys])
        return pd.DataFrame(rows, columns=["anchor"] + keys)

    def as_compact_dict(self):
        me_as_dict = self.as_dataframe().to_dict(orient="list")
        for metric in list(me_as_dict.keys()):
            values = me_as_dict[metric]
            has_array_vals = isinstance(values[0], np.ndarray)
            if has_array_vals:
                me_as_dict[metric] = [v.tolist() for v in values]
        return me_as_dict

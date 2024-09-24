from itertools import combinations, product

import numpy as np
from .timer import Timer

from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)


class ClassificationScorer:
    def __init__(self, classes_learner: list, classes_overall: list, timer: Timer = None) -> None:
        if not isinstance(classes_learner, list):
            raise ValueError(f"'classes_learner' must be a list but is {type(classes_learner)}")
        if not isinstance(classes_overall, list):
            raise ValueError(f"'classes_overall' must be a list but is {type(classes_overall)}")
        self.classes_learner = classes_learner
        self.classes_overall = classes_overall
        self.padded_classes = sorted(set(classes_overall) - set(classes_learner))
        if len(self.padded_classes) > 0:
            label_order_after_expansion = self.classes_learner + self.padded_classes
            self.reordering_index = [label_order_after_expansion.index(label) for label in self.classes_overall]
        else:
            self.reordering_index = None
        self.timer = Timer() if timer is None else timer

    def score(self, y_true, y_pred, y_pred_proba):

        # make sure that y_predict_proba is a matrix over all known labels (not only the ones known to the learner)
        if len(self.padded_classes) > 0:
            expansion_matrix = np.zeros(
                (len(y_true), len(self.padded_classes))
            )
            y_pred_proba = np.concatenate([y_pred_proba, expansion_matrix], axis=1)
            y_pred_proba = y_pred_proba[:, self.reordering_index]

        labels_in_ground_truth = sorted(np.unique(y_true))
        num_labels_in_ground_truth = len(labels_in_ground_truth)
        is_unary = num_labels_in_ground_truth == 1
        is_binary = num_labels_in_ground_truth == 2

        metric_names = [
            "confusion_matrix",
            "auc",
            "log_loss",
            "brier_score",
        ]

        scores = {}

        for metric_name in metric_names:
            with self.timer.time(metric_name) as metric_timer:
                score = None

                if metric_name == "confusion_matrix":
                    score = np.round(
                        confusion_matrix(y_true, y_pred, labels=self.classes_overall), 5
                    ).tolist()

                elif metric_name == "auc":
                    if is_unary:
                        score = np.nan  # AUC not defined for single class problems
                    elif is_binary:
                        score = np.round(
                            roc_auc_score(
                                y_true, y_pred_proba[:, 1], labels=self.classes_overall
                            ),
                            5,
                        )
                    else:

                        # if the learner has a superset of the ground truth labels, only use the ground truth labels
                        if set(y_true).issubset(set(self.classes_learner)):
                            accepted_labels = labels_in_ground_truth

                        # otherwise use all the labels that are either present in ground truth or in the predictions
                        else:
                            accepted_labels = [
                                l for i, l in enumerate(self.classes_overall)
                                if (
                                    l in y_true or
                                    y_pred_proba[:, i].sum() > 0
                                )
                            ]

                        # generate a proper distribution over the remaining labels
                        mask_labels_auc = np.isin(self.classes_overall, accepted_labels)
                        y_pred_proba_auc = y_pred_proba[:, mask_labels_auc]
                        if num_labels_in_ground_truth != len(self.classes_learner):
                            y_pred_proba_auc /= y_pred_proba_auc.sum(axis=1, keepdims=1)

                        # compute the different AUC scores
                        score = {}
                        for multi_class, average in product(
                            ["ovr", "ovo"], ["micro", "macro", "weighted", None]
                        ):
                            if average in [None, "micro"] and multi_class != "ovr":
                                continue
                            if np.any(np.isnan(y_pred_proba_auc)):
                                auc = np.nan
                            else:
                                try:
                                    auc = np.round(
                                        roc_auc_score(
                                            y_true=y_true,
                                            y_score=y_pred_proba_auc,
                                            labels=accepted_labels,
                                            multi_class=multi_class,
                                            average=average,
                                        ),
                                        5,
                                    )
                                except ValueError as e:
                                    if "Only one class present in y_true." in str(e):
                                        auc = np.nan
                                    else:
                                        raise e

                            score[f"auc_{multi_class}_{average}"] = auc
                elif metric_name == "log_loss":
                    y_base = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
                    score = np.round(
                        log_loss(y_true, y_base, labels=self.classes_overall), 5
                    )
                elif metric_name == "brier_score":
                    if is_binary:
                        score = np.round(
                            brier_score_loss(
                                y_true, y_pred_proba[:, 1], pos_label=self.classes_overall[1]
                            ),
                            5,
                        )
                    else:
                        y_true_binarized = np.zeros((len(y_true), len(self.classes_overall)))
                        for j, label in enumerate(self.classes_overall):
                            mask = y_true == label
                            y_true_binarized[mask, j] = 1
                        score = np.round(
                            ((y_true_binarized - y_pred_proba) ** 2).sum(axis=1).mean(),
                            5,
                        )

                metric_timer["value"] = score

                scores[metric_name] = score
        return scores


class RegressionScorer:
    def __init__(self, timer: Timer = None) -> None:
        self.timer = Timer() if timer is None else timer

    def score(self, y_true, y_pred):
        metric_names = ["r2", "mean_squared_error"]

        scores = {}

        for metric_name in metric_names:
            with self.timer.time(metric_name) as metric_timer:
                score = None

                if metric_name == "r2":
                    score = np.round(mean_squared_error(y_true, y_pred), 5).tolist()
                elif metric_name == "mean_squared_error":
                    score = np.round(r2_score(y_true, y_pred), 5).tolist()

                metric_timer["value"] = score
                scores[metric_name] = score

        return scores

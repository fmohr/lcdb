import itertools as it

import numpy as np
from lcdb.timer import Timer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


class ClassificationScorer:
    def __init__(self, classes: list, timer: Timer = None) -> None:
        if not isinstance(classes, list):
            raise ValueError(f"'classes'] must be a list but is {type(classes)}")
        self.classes = classes
        self.timer = Timer() if timer is None else timer

    def score(self, y_true, y_pred, y_pred_proba):
        relevant_labels = self.classes.copy()

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

        metric_names = [
            "confusion_matrix",
            # TODO: Removed because redundant with confusion_matrix
            # "accuracy",
            "auc",
            "log_loss",
            "brier_score",
        ]
        
        scores = {}

        for metric_name in metric_names:
            with self.timer.time(metric_name) as metric_timer:
                if metric_name == "confusion_matrix":
                    score = np.round(
                        confusion_matrix(y_true, y_pred, labels=relevant_labels), 5
                    ).tolist()

                elif metric_name == "accuracy":
                    # TODO: why not balanced accuracy?
                    score = np.round(accuracy_score(y_true, y_pred), 5)
                elif metric_name == "auc":
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
                elif metric_name == "log_loss":
                    y_base = y_pred_proba[:, 1] if is_binary else y_pred_proba
                    score = np.round(
                        log_loss(y_true, y_base, labels=relevant_labels), 5
                    )
                elif metric_name == "brier":
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
                            ((y_true_binarized - y_pred_proba) ** 2).sum(axis=1).mean(),
                            5,
                        )

                metric_timer["value"] = score
                
                scores[metric_name] = score

        return scores 

import functools
import logging
import traceback

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from lcdb.data.split import train_valid_test_split
import warnings
from lcdb.timer import Timer
from lcdb.utils import (
    terminate_on_timeout,
    FunctionCallTimeoutError,
    get_anchor_schedule
)

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, confusion_matrix, log_loss, accuracy_score
import warnings
import itertools as it


class LCController:

    def __init__(self,
                 workflow,
                 X,
                 y,
                 dataset_metadata,
                 test_seed,
                 valid_seed,
                 valid_prop: float = 0.1,
                 test_prop: float = 0.1,
                 stratify=True,
                 monotonic=False,
                 timeout_on_fit=-1,
                 known_categories: bool = True,
                 raise_errors: bool = False
                 ):

        self.workflow = workflow

        self.num_instances = X.shape[0]
        self.dataset_metadata = dataset_metadata

        self.X = X
        self.y = y
        self.labels = list(np.unique(y))
        self.is_binary = len(self.labels) == 2
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = train_valid_test_split(
            X, y, test_seed, valid_seed, test_prop, valid_prop, stratify=True
        )
        self.valid_seed = valid_seed
        self.test_seed = test_seed
        self.monotonic = monotonic
        self.timeout_on_fit = timeout_on_fit
        self.raise_errors = raise_errors
        self.anchors = get_anchor_schedule(int(self.num_instances * (1 - test_prop - valid_prop)))

        # state variables
        self.cur_anchor = None
        self.X_train_at_anchor = None
        self.y_train_at_anchor = None
        self.labels_as_used_by_workflow = None  # list of labels, this order is defined by the workflow
        self.timer = None  # we use a different timer for each anchor

        # Transform categorical features
        columns_categories = np.asarray(dataset_metadata["categories"], dtype=bool)
        values_categories = None

        dataset_metadata["categories"] = {"columns": columns_categories}
        if not (np.any(columns_categories)):
            one_hot_encoder = FunctionTransformer(func=lambda x: x, validate=False)
        else:
            dataset_metadata["categories"]["values"] = None
            one_hot_encoder = OneHotEncoder(
                drop="first", sparse_output=False
            )  # TODO: drop "first" could be an hyperparameter
            one_hot_encoder.fit(X[:, columns_categories])
            if known_categories:
                values_categories = one_hot_encoder.categories_
                values_categories = [v.tolist() for v in values_categories]
                dataset_metadata["categories"]["values"] = values_categories

        # create report
        self.report = {
            "workflow": workflow.__class__.__name__,
            "valid_prop": valid_prop,
            "test_prop": valid_prop,
            "monotonic": monotonic,
            "valid_seed": valid_seed,
            "test_seed": test_seed,
            "fidelity_unit": "samples",
            "fidelity_values": [],
            "scores": [],
            "times": [],
            "child_fidelities": []
        }

    def set_anchor(self, anchor):

        train_idx = np.arange(self.X_train.shape[0])
        # If not monotonic, the training set should be shuffled differently for each anchor
        # so that the training sets of different anchors do not contain eachother
        i = self.anchors.index(anchor)
        if not self.monotonic:
            random_seed_train_shuffle = np.random.RandomState(self.valid_seed).randint(
                0, 2**32 - 1, size=len(self.anchors)
            )[i]
            rs = np.random.RandomState(random_seed_train_shuffle)
            rs.shuffle(train_idx)

            X_train = self.X_train[train_idx]
            y_train = self.y_train[train_idx]
        else:
            X_train, y_train = self.X_train, self.y_train

        self.cur_anchor = anchor
        self.X_train_at_anchor = X_train[:anchor]
        self.y_train_at_anchor = y_train[:anchor]

    def build_curves(self):

        # Build sample-wise learning curve
        for anchor in self.anchors:

            self.set_anchor(anchor)
            self.timer = Timer()
            self.workflow.timer = self.timer
            logging.info(
                f"Running anchor {anchor} which is {anchor / self.X_train_at_anchor.shape[0] * 100:.2f}% of the dataset."
            )
            self.fit_workflow_on_current_anchor()

            # Predict and Score
            logging.info("Predicting and scoring...")
            scores = self.compute_metrics_for_workflow()

            # Collect Infos
            self.report["fidelity_values"].append(anchor)
            self.report["scores"].append(scores)
            self.report["times"].append(self.timer.runtimes)

            # Collect child fidelities (e.g., iteration learning curves with epochs) if available
            if hasattr(self.workflow, "fidelities"):
                self.report["child_fidelities"].append(self.workflow.fidelities)

    def fit_workflow_on_current_anchor(self):
        if self.timeout_on_fit > 0:
            self.workflow.fit = functools.partial(
                self.terminate_on_timeout, self.timeout_on_fit, self.workflow.fit
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                if self.workflow.requires_valid_to_fit:
                    if self.workflow.requires_test_to_fit:
                        self.workflow.fit(
                            self.X_train_at_anchor,
                            self.y_train_at_anchor,
                            X_valid=self.X_valid,
                            y_valid=self.y_valid,
                            X_test=self.X_test,
                            y_test=self.y_test,
                            metadata=self.dataset_metadata,
                        )
                    else:
                        self.workflow.fit(
                            self.X_train_at_anchor,
                            self.y_train_at_anchor,
                            X_valid=self.X_valid,
                            y_valid=self.y_valid,
                            metadata=self.dataset_metadata,
                        )
                else:
                    if self.workflow.requires_test_to_fit:
                        self.workflow.fit(
                            self.X_train_at_anchor,
                            self.y_train_at_anchor,
                            X_test=self.X_test,
                            y_test=self.y_test,
                            metadata=self.dataset_metadata,
                        )
                    else:
                        self.workflow.fit(self.X_train_at_anchor, self.y_train_at_anchor, metadata=self.dataset_metadata)

            except Exception as exception:
                if self.raise_errors:
                    raise

                # Collect child fidelities (e.g., iteration learning curves with epochs) if available
                if hasattr(self.workflow, "fidelities"):
                    self.report["child_fidelities"].append(self.workflow.fidelities)

                self.report["traceback"] = traceback.format_exc()

                logging.error(
                    f"Error while fitting the workflow: \n{self.report['traceback']}"
                )

                self.report["traceback"] = r'"{}"'.format(self.report["traceback"])

                # The evaluation is considered a total failure only if
                # None of the anchors returned scored.
                if (
                    len(self.report["scores"]) > 0
                    and len(self.report["scores"][-1]) > 0
                ):
                    # -1: last fidelity, 1: validation set, 0: accuracy
                    objective = self.report["scores"][-1][1][0]
                else:
                    objective = "F"

                    if isinstance(exception, FunctionCallTimeoutError):
                        objective += "_function_call_timeout_error"
                    elif isinstance(exception, MemoryError):
                        objective += "_memory_error"

    def get_predictions(self, fitted_workflow):
        keys = {}
        labels = fitted_workflow.infos["classes_"]

        for X_, y_true, postfix in [
            (self.X_train_at_anchor, self.y_train_at_anchor, "train"),
            (self.X_valid, self.y_valid, "val"),
            (self.X_test, self.y_test, "test")]:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_workflow.timer.enter(postfix)
                keys[f"y_pred_{postfix}"] = fitted_workflow.predict(X_)
                keys[f"y_pred_proba_{postfix}"] = fitted_workflow.predict_proba(X_)
                fitted_workflow.timer.leave()
        return keys, labels

    def compute_metrics_for_workflow(self):
        predictions, labels = self.get_predictions(self.workflow)
        self.labels_as_used_by_workflow = labels
        return self.compute_metrics_from_predictions(**predictions)

    def compute_metrics_from_predictions(self,
            y_pred_train,
            y_pred_proba_train,
            y_pred_val,
            y_pred_proba_val,
            y_pred_test,
            y_pred_proba_test):
        scores = {}

        for y_true, y_pred, y_pred_proba, postfix in [
            (self.y_train_at_anchor, y_pred_train, y_pred_proba_train, "train"),
            (self.y_valid, y_pred_val, y_pred_proba_val, "val"),
            (self.y_test, y_pred_test, y_pred_proba_test, "test")
        ]:

            relevant_labels = list(self.labels_as_used_by_workflow.copy())
            labels_in_true_data_not_used_by_workflow = list(set(y_true).difference(relevant_labels))
            if len(labels_in_true_data_not_used_by_workflow) > 0:
                expansion_matrix = np.zeros((len(y_true), len(labels_in_true_data_not_used_by_workflow)))
                relevant_labels.extend(labels_in_true_data_not_used_by_workflow)
                y_pred_proba = np.column_stack([y_pred_proba, expansion_matrix])
                y_pred_proba = np.column_stack([y_pred_proba[:, i] for i in np.argsort(relevant_labels)])
                relevant_labels = sorted(relevant_labels)

            self.timer.enter(postfix)
            for target in ["cm", "accuracy", "auc", "ll", "bl"]:

                self.timer.start(f"metric_{target}")
                if target == "cm":
                    score = np.round(confusion_matrix(y_true, y_pred, labels=relevant_labels), 5)
                elif target == "accuracy":
                    score = np.round(accuracy_score(y_true, y_pred), 5)
                elif target == "auc":
                    if self.is_binary:
                        score = np.round(roc_auc_score(y_true, y_pred_proba[:, 1], labels=relevant_labels), 5)
                    else:
                        score = {}
                        for multi_class, average in it.product(["ovr", "ovo"], ["micro", "macro", "weighted", None]):
                            if average in [None, "micro"] and multi_class != "ovr":
                                continue
                            auc = np.round(
                                roc_auc_score(
                                    y_true,
                                    y_pred_proba,
                                    labels=relevant_labels,
                                    multi_class=multi_class,
                                    average=average
                                ), 5)
                            score[f"{multi_class}_{average}"] = auc
                elif target == "ll":
                    y_base = y_pred_proba[:, 1] if self.is_binary else y_pred_proba
                    score = np.round(log_loss(y_true, y_base, labels=relevant_labels), 5)
                elif target == "bl":
                    if self.is_binary:
                        score = np.round(brier_score_loss(y_true, y_pred_proba[:, 1], pos_label=relevant_labels[1]), 5)
                    else:
                        y_true_binarized = np.zeros((len(y_true), len(relevant_labels)))
                        for j, label in enumerate(relevant_labels):
                            mask = y_true == label
                            y_true_binarized[mask, j] = 1
                        score = np.round(((y_true_binarized - y_pred_proba) ** 2).sum(axis=1).mean(), 5)

                # store results and time
                self.timer.stop(f"metric_{target}")
                scores[f"{target}_{postfix}"] = score

        return scores

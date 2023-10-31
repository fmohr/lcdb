import functools
import logging
import traceback

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from lcdb.data.split import train_valid_test_split
import warnings
from lcdb.timer import Timer
from lcdb.Curve import Curve
from lcdb.CurveDB import CurveDB
from lcdb.utils import (
    terminate_on_timeout,
    FunctionCallTimeoutError,
    get_anchor_schedule
)

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, confusion_matrix, log_loss, accuracy_score
import warnings
import pandas as pd


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
        self.timer = Timer(precision=6)
        self.workflow.timer = self.timer  # overwrite timer of workflow

        # state variables
        self.cur_anchor = None
        self.X_train_at_anchor = None
        self.y_train_at_anchor = None
        self.labels_as_used_by_workflow = None  # list of labels, this order is defined by the workflow
        self.curves = None
        self.additional_data_per_anchor = None

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
            "test_seed": test_seed
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
        self.curves = {
            "train": Curve(workflow=self.workflow, timer=self.timer),
            "val": Curve(workflow=self.workflow, timer=self.timer),
            "test": Curve(workflow=self.workflow, timer=self.timer)
        }
        self.additional_data_per_anchor = {}

        for anchor in self.anchors:

            self.set_anchor(anchor)
            self.timer.enter(anchor)
            logging.info(
                f"Running anchor {anchor} which is {anchor / self.X_train_at_anchor.shape[0] * 100:.2f}% of the dataset."
            )
            self.fit_workflow_on_current_anchor()

            # Collect the fit report (e.g., with iteration learning curves with epochs) if available
            if hasattr(self.workflow, "fit_report"):
                self.additional_data_per_anchor[anchor].update(self.workflow.fit_report)

            # Predict and Score
            logging.info("Predicting and scoring...")
            self.compute_metrics_for_workflow()
            self.timer.leave()

        self.report["curve_db"] = CurveDB(
            self.curves["train"],
            self.curves["val"],
            self.curves["test"],
            self.timer.runtimes,
            self.additional_data_per_anchor
        ).dump_to_dict()

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
        labels = fitted_workflow.infos["classes"]

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
        return self.extend_curves_based_on_predictions(**predictions)

    def extend_curves_based_on_predictions(self,
                                           y_pred_train,
                                           y_pred_proba_train,
                                           y_pred_val,
                                           y_pred_proba_val,
                                           y_pred_test,
                                           y_pred_proba_test):

        for y_true, y_pred, y_pred_proba, postfix in [
            (self.y_train_at_anchor, y_pred_train, y_pred_proba_train, "train"),
            (self.y_valid, y_pred_val, y_pred_proba_val, "val"),
            (self.y_test, y_pred_test, y_pred_proba_test, "test")
        ]:
            self.timer.enter(postfix)
            curve = self.curves[postfix]
            curve.compute_metrics(self.cur_anchor, y_true, y_pred, y_pred_proba)
            self.timer.leave()

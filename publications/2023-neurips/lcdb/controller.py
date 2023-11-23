import functools
import itertools as it
import logging
import pprint
import traceback
import warnings

import numpy as np
from lcdb.data.split import train_valid_test_split
from lcdb.timer import Timer
from lcdb.utils import (
    FunctionCallTimeoutError,
    get_anchor_schedule,
    terminate_on_timeout,
)
from lcdb.scorer import ClassificationScorer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


class LCController:
    def __init__(
        self,
        timer: Timer,
        workflow_factory,
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
        raise_errors: bool = False,
    ):
        self.timer = timer
        self.workflow_factory = workflow_factory
        self.workflow = None

        self.num_instances = X.shape[0]
        self.dataset_metadata = dataset_metadata

        self.X = X
        self.y = y
        self.labels = list(np.unique(y))
        self.is_binary = len(self.labels) == 2
        (
            self.X_train,
            self.X_valid,
            self.X_test,
            self.y_train,
            self.y_valid,
            self.y_test,
        ) = train_valid_test_split(
            X, y, test_seed, valid_seed, test_prop, valid_prop, stratify=stratify
        )
        self.valid_seed = valid_seed
        self.test_seed = test_seed
        self.monotonic = monotonic
        self.timeout_on_fit = timeout_on_fit
        self.raise_errors = raise_errors
        self.anchors = get_anchor_schedule(
            int(self.num_instances * (1 - test_prop - valid_prop))
        )

        # state variables
        self.cur_anchor = None
        self.X_train_at_anchor = None
        self.y_train_at_anchor = None
        self.labels_as_used_by_workflow = (
            None  # list of labels, this order is defined by the workflow
        )
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
            "valid_prop": valid_prop,
            "test_prop": valid_prop,
            "monotonic": monotonic,
            "valid_seed": valid_seed,
            "test_seed": test_seed,
            "traceback": None,
        }

        self.objective = None

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

        with self.timer.time("build_curves"):
            for anchor in self.anchors:
                self.set_anchor(anchor)

                with self.timer.time("anchor", {"value": anchor}) as anchor_timer:
                    logging.info(
                        f"Running anchor {anchor} which is {anchor / self.X_train.shape[0] * 100:.2f}% of the dataset."
                    )

                    error_code = self.fit_workflow_on_current_anchor()

                    if error_code != 0:
                        # Cancel timers that were started in fit_workflow_on_current_anchor
                        self.timer.cancel(anchor_timer.id, only_children=True)

                    assert (
                        self.timer.active_node.id == anchor_timer.id
                    ), f"The active timer is not correct, it is {self.timer.active_node} when it should be {anchor_timer} "

                    # If an error was detected then skip scoring...
                    if error_code != 0:
                        break

                    # Predict and Score
                    logging.info("Predicting and scoring...")
                    try:
                        self.compute_metrics_for_workflow()
                    except Exception as exception:
                        
                        # Cancel timers that were started in 'try' block
                        self.timer.cancel(anchor_timer.id, only_children=True)

                        # Collect traceback
                        self.report["traceback"] = traceback.format_exc()

                        logging.error(
                            f"Error while fitting the workflow: \n{self.report['traceback']}"
                        )

                        self.report["traceback"] = r'"{}"'.format(
                            self.report["traceback"]
                        )

                        # The evaluation is considered a total failure only if
                        # None of the anchors returned scored.
                        if self.objective is None:
                            self.objective = "F"

                            if isinstance(exception, ValueError):
                                self.objective += "_value_error"
                        error_code = 1
                        break

    def fit_workflow_on_current_anchor(self) -> int:
        """Fit the workflow on the current anchor.

        Returns 0 if the workflow was fitted successfully, 1 otherwise.
        """

        # Represent success (0) or failure (1) while fitting the workflow
        error_code = 0

        with self.timer.time("create_workflow"):
            self.workflow = self.workflow_factory()

        if self.timeout_on_fit > 0:
            self.workflow.fit = functools.partial(
                terminate_on_timeout, self.timeout_on_fit, self.workflow.fit
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

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
                        self.workflow.fit(
                            self.X_train_at_anchor,
                            self.y_train_at_anchor,
                            metadata=self.dataset_metadata,
                        )

        except Exception as exception:
            self.report["traceback"] = traceback.format_exc()

            logging.error(
                f"Error while fitting the workflow: \n{self.report['traceback']}"
            )

            self.report["traceback"] = r'"{}"'.format(self.report["traceback"])

            # The evaluation is considered a total failure only if
            # None of the anchors returned scored.
            if self.objective is None:
                self.objective = "F"

                if isinstance(exception, FunctionCallTimeoutError):
                    self.objective += "_function_call_timeout_error"
                elif isinstance(exception, MemoryError):
                    self.objective += "_memory_error"

            error_code = 1

        return error_code

    def compute_metrics_for_workflow(self):
        predictions, labels = self.get_predictions()
        self.labels_as_used_by_workflow = labels
        return self.score_predictions(**predictions)

    def get_predictions(self):
        keys = {}
        labels = self.workflow.infos["classes"]

        with self.timer.time("get_predictions"):
            for X_split, label_split in [
                (self.X_train_at_anchor, "train"),
                (self.X_valid, "val"),
                (self.X_test, "test"),
            ]:
                with warnings.catch_warnings(), self.timer.time(label_split):
                    warnings.simplefilter("ignore")

                    # TODO: this should be replaced to avoid infering twice
                    keys[f"y_pred_{label_split}"] = self.workflow.predict(X_split)
                    keys[f"y_pred_proba_{label_split}"] = self.workflow.predict_proba(
                        X_split
                    )

        return keys, labels

    def score_predictions(
        self,
        y_pred_train,
        y_pred_proba_train,
        y_pred_val,
        y_pred_proba_val,
        y_pred_test,
        y_pred_proba_test,
    ):
        scorer = ClassificationScorer(
            classes=self.workflow.infos["classes"], timer=self.timer
        )

        with self.timer.time("metrics"):
            for y_true, y_pred, y_pred_proba, label_split in [
                (self.y_train_at_anchor, y_pred_train, y_pred_proba_train, "train"),
                (self.y_valid, y_pred_val, y_pred_proba_val, "val"),
                (self.y_test, y_pred_test, y_pred_proba_test, "test"),
            ]:
                with self.timer.time(label_split) as split_timer:
                    scores = scorer.score(y_true, y_pred, y_pred_proba)

                    if label_split == "val":
                        self.objective = -scores["log_loss"]

        return 0  # no error occurred

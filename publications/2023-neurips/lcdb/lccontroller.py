import functools
import logging
import traceback
import warnings

import numpy as np
from lcdb.curve import Curve
from lcdb.curvedb import CurveDB
from lcdb.data.split import train_valid_test_split
from lcdb.timer import Timer
from lcdb.utils import (
    FunctionCallTimeoutError,
    get_anchor_schedule,
    terminate_on_timeout,
)
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


class LCController:
    def __init__(
        self,
        workflow,
        X,
        y,
        dataset_metadata,
        test_seed,
        valid_seed,
        valid_prop: float = 0.1,
        test_prop: float = 0.1,
        stratify=True,  # TODO: remove if not used
        monotonic=False,
        timeout_on_fit=-1,
        known_categories: bool = True,
        raise_errors: bool = False,
    ):
        self.workflow = workflow

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
            X, y, test_seed, valid_seed, test_prop, valid_prop, stratify=True
        )
        self.valid_seed = valid_seed
        self.test_seed = test_seed
        self.monotonic = monotonic
        self.timeout_on_fit = timeout_on_fit
        self.raise_errors = raise_errors
        self.anchors = get_anchor_schedule(
            int(self.num_instances * (1 - test_prop - valid_prop))
        )
        self.workflow.timer = Timer(precision=6)  # overwrite timer of workflow

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
            "workflow": workflow.__class__.__name__,
            "valid_prop": valid_prop,
            "test_prop": valid_prop,
            "monotonic": monotonic,
            "valid_seed": valid_seed,
            "test_seed": test_seed,
            "traceback": None,
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
            "train": Curve(workflow=self.workflow, timer=self.workflow.timer),
            "val": Curve(workflow=self.workflow, timer=self.workflow.timer),
            "test": Curve(workflow=self.workflow, timer=self.workflow.timer),
        }
        self.additional_data_per_anchor = {}

        self.workflow.timer.start("curvecomputations")
        for anchor in self.anchors:
            self.set_anchor(anchor)
            self.workflow.timer.start(anchor)
            timer_stack_size = len(self.workflow.timer.stack)
            logging.info(
                f"Running anchor {anchor} which is {anchor / self.X_train_at_anchor.shape[0] * 100:.2f}% of the dataset."
            )
            error_code = self.fit_workflow_on_current_anchor()
            assert len(self.workflow.timer.stack) == timer_stack_size, (
                f"The timer stack has more elements than expected. You forgot to stop a started timer. "
                f"Active timers: {self.workflow.timer.get_simplified_stack()}"
            )

            # Collect the fit report (e.g., with iteration learning curves with epochs) if available
            if hasattr(self.workflow, "fit_report"):
                self.additional_data_per_anchor[anchor] = self.workflow.fit_report

            # Predict and Score
            if error_code == 0:
                logging.info("Predicting and scoring...")
                error_code = self.compute_metrics_for_workflow()

                # Set objective
                if error_code == 0:
                    self.objective = self.curves["val"][self.curves["val"].anchors[-1]][
                        "accuracy"
                    ]

            # stop timer for activities at this anchor
            self.workflow.timer.stop()

        self.workflow.timer.stop()  # outmost level
        assert (
            len(self.workflow.timer.stack) == 0
        ), "The timer stack is not empty. You forgot to stop a started timer."
        self.report["curve_db"] = CurveDB(
            self.curves["train"],
            self.curves["val"],
            self.curves["test"],
            self.workflow.timer.get_simplified_dict(
                multiple_occurrences="merge_and_drop"
            )["children"],
            self.additional_data_per_anchor,
        ).dump_to_dict()

    def fit_workflow_on_current_anchor(self) -> int:
        """Fit the workflow on the current anchor.

        Returns 0 if the workflow was fitted successfully, 1 otherwise.
        """

        # Represent success (0) or failure (1) while fitting the workflow
        error_code = 0

        if self.timeout_on_fit > 0:
            self.workflow.fit = functools.partial(
                terminate_on_timeout, self.timeout_on_fit, self.workflow.fit
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
                        self.workflow.fit(
                            self.X_train_at_anchor,
                            self.y_train_at_anchor,
                            metadata=self.dataset_metadata,
                        )

            except Exception as exception:
                # make sure that nothing related to fit is on the timer stack anymore
                fit_is_on_timer_stack = np.any(
                    [e["name"] == "fit" for e in self.workflow.timer.stack]
                )
                if fit_is_on_timer_stack:
                    while self.workflow.timer.stack[-1]["name"] != "fit":
                        self.workflow.timer.stop()
                    self.workflow.timer.stop()

                if self.raise_errors:
                    raise

                self.report["traceback"] = traceback.format_exc()

                logging.error(
                    f"Error while fitting the workflow: \n{self.report['traceback']}"
                )

                self.report["traceback"] = r'"{}"'.format(self.report["traceback"])

                # The evaluation is considered a total failure only if
                # None of the anchors returned scored.
                if len(self.curves["val"].anchors) == 0:
                    self.objective = "F"

                    if isinstance(exception, FunctionCallTimeoutError):
                        self.objective += "_function_call_timeout_error"
                    elif isinstance(exception, MemoryError):
                        self.objective += "_memory_error"

                error_code = 1

            return error_code

    def get_predictions(self, fitted_workflow):
        keys = {}
        labels = fitted_workflow.infos["classes"]

        for X_, y_true, postfix in [
            (self.X_train_at_anchor, self.y_train_at_anchor, "train"),
            (self.X_valid, self.y_valid, "val"),
            (self.X_test, self.y_test, "test"),
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_workflow.timer.start(postfix)
                try:
                    # TODO: this should be replaced to avoid infering twice
                    keys[f"y_pred_{postfix}"] = fitted_workflow.predict(X_)
                    keys[f"y_pred_proba_{postfix}"] = fitted_workflow.predict_proba(X_)
                except KeyboardInterrupt:
                    raise
                except:
                    while self.workflow.timer.stack[-1]["name"] != postfix:
                        self.workflow.timer.stop()  # stop the timer for predictions
                    self.workflow.timer.stop()  # stop the timer for train/val/test
                    raise
                fitted_workflow.timer.stop()
        return keys, labels

    def compute_metrics_for_workflow(self):
        try:
            predictions, labels = self.get_predictions(self.workflow)
            self.labels_as_used_by_workflow = labels
            return self.extend_curves_based_on_predictions(**predictions)
        except KeyboardInterrupt:
            raise
        except:
            print("Failure in prediction making, not computing any metrics ...")
            return 1

    def extend_curves_based_on_predictions(
        self,
        y_pred_train,
        y_pred_proba_train,
        y_pred_val,
        y_pred_proba_val,
        y_pred_test,
        y_pred_proba_test,
    ):
        for y_true, y_pred, y_pred_proba, postfix in [
            (self.y_train_at_anchor, y_pred_train, y_pred_proba_train, "train"),
            (self.y_valid, y_pred_val, y_pred_proba_val, "val"),
            (self.y_test, y_pred_test, y_pred_proba_test, "test"),
        ]:
            self.workflow.timer.start(postfix)
            curve = self.curves[postfix]
            curve.compute_metrics(self.cur_anchor, y_true, y_pred, y_pred_proba)
            self.workflow.timer.stop()
        return 0  # no error occurred
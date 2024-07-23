import sklearn
from ConfigSpace import (
    Constant,
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
    EqualsCondition,
)
from ...experiments.scorer import ClassificationScorer
from ...experiments.utils import get_schedule 

import numpy as np
import time

from ._base import SklearnWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.BaggingWorkflow",
    space={
        "n_estimators": Constant("n_estimators", value=32),
        "criterion": Categorical(
            "criterion", items=["gini", "entropy", "log_loss"], default="gini"
        ),
        "min_samples_split": Integer("min_samples_split", bounds=(2, 50), default=2),
        "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 25), default=2),
        "max_features": Categorical(
            "max_features", items=["all", "sqrt", "log2"], default="sqrt"
        ),
        "min_impurity_decrease": Float(
            "min_impurity_decrease", bounds=(0.0, 1.0), default=0.0
        ),
        "bootstrap": Categorical("bootstrap", items=[True, False], default=True),
        "max_samples": Float(
            "max_samples", bounds=(10**-3, 1.0), default=1.0
        ),  # cannot be 0
    },
)

CONFIG_SPACE.add_condition(
    EqualsCondition(CONFIG_SPACE["max_samples"], CONFIG_SPACE["bootstrap"], True)
)


class BaggingWorkflow(SklearnWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        bagging,
        n_estimators,
        timer=None,
        epoch_schedule: str = "full",
        iterations_to_wait_for_update=4,
        **kwargs,
    ):
        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

        self.max_n_estimators = n_estimators
        self.iterations_to_wait_for_update = iterations_to_wait_for_update
        self.last_recorded_iteration = None

        super().__init__(learner=bagging, timer=timer, **kwargs)

        # Scoring Schedule for Sub-fidelity
        self.schedule = get_schedule(
            name=epoch_schedule, n=self.max_n_estimators  # , base=2, power=0.5, delay=0
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(
        self, X, y, X_valid, y_valid, X_test, y_test, metadata
    ):
        self.metadata = metadata

        # first train full bagging ensemble. This is because training them iteratively is highly inefficient in sklearn
        ts_train_start = time.time()
        self.learner.set_params(n_estimators=self.max_n_estimators)
        self.learner.fit(X, y)
        ts_train_stop = time.time()

        # TODO: UNUSED VARIABLE?
        avg_fit_time_per_learner = (
            ts_train_stop - ts_train_start
        ) / self.learner.n_estimators

        data = dict(
            train=dict(X=X, y=y),
            val=dict(X=X_valid, y=y_valid),
            test=dict(X=X_test, y=y_test),
        )

        # set up stuff for OOB measurements
        n_samples = X.shape[0]
        n_samples_bootstrap = sklearn.ensemble._forest._get_n_samples_bootstrap(
            n_samples,
            self.learner.max_samples,
        )
        get_unsampled_indices = (
            lambda tree: sklearn.ensemble._forest._generate_unsampled_indices(
                tree.random_state,
                n_samples,
                n_samples_bootstrap,
            )
        )

        scorer = ClassificationScorer(
            classes_learner=list(self.learner.classes_),
            classes_overall=self.infos["classes_overall"],
            timer=self.timer,
        )
        y_pred_proba_oob = np.zeros((n_samples, len(np.unique(y))))

        # now compute metrics for partial forest sizes (and simulate the time for training)
        for i, n_estimators in enumerate(self.schedule):

            with self.timer.time("epoch", metadata={"value": n_estimators}):
                with self.timer.time("epoch_train"):

                    # TODO: ADD SYNTHETIC DELAY TO TIMER
                    pass

                # compute test scores
                with self.timer.time("epoch_test"):
                    with self.timer.time("metrics"):
                        # compute train, validation, and test scores of current forest
                        for label_split, data_split in data.items():
                            with self.timer.time(label_split):
                                with self.timer.time("predict_with_proba"):
                                    (
                                        y_pred,
                                        y_pred_proba,
                                    ) = self._predict_with_proba_without_transform(
                                        data_split["X"], n_estimators=n_estimators
                                    )

                                # TODO: This could be optimized to only get predictions for the last tree. As for OOB
                                y_true = data_split["y"]
                                scorer.score(
                                    y_true=y_true,
                                    y_pred=y_pred,
                                    y_pred_proba=y_pred_proba,
                                )

                        if self.learner.bootstrap:

                            # compute train, validation, and test scores of current forest
                            counters = np.zeros(y_pred_proba_oob.shape)
                            with self.timer.time("oob") as oob_timer:
                                start_index = self.schedule[i - 1] if i > 0 else 0
                                labels = list(self.learner.classes_)
                                n_labels = len(labels)
                                for t in range(start_index, n_estimators):
                                    considered_tree = self.learner.estimators_[t]
                                    val_indices = get_unsampled_indices(considered_tree)
                                    y_pred_oob_tree = considered_tree.predict_proba(
                                        X[val_indices]
                                    )
                                    y_pred_proba_oob[val_indices] = (
                                        y_pred_proba_oob[val_indices]
                                        * counters[val_indices]
                                        + y_pred_oob_tree
                                    ) / (counters[val_indices] + 1)
                                    counters[val_indices] += 1

                                y_pred_oob = np.array(
                                    [
                                        labels[ind]
                                        for ind in np.argmax(y_pred_proba_oob, axis=1)
                                    ]
                                )
                                y_true = y

                                # Only keep OOB samples
                                sum_of_probs = y_pred_proba_oob.sum(axis=1)
                                mask = sum_of_probs > 0.0
                                num_samples = np.sum(mask)
                                oob_timer["num_samples"] = float(num_samples)
                                assert np.allclose(
                                    1, sum_of_probs[mask]
                                ), f"NOT A DISTRIBUTION: {y_pred_proba_oob[mask]}"

                                scorer.score(
                                    y_true=y_true[mask],
                                    y_pred=y_pred_oob[mask],
                                    y_pred_proba=y_pred_proba_oob[mask],
                                )

        self.infos["classes"] = list(self.learner.classes_)

    def _predict_after_transform(self, X):
        return self.learner.predict(X)

    def _predict_proba_after_transform(self, X, n_estimators=None):

        # partially take out classifiers if necessary
        if n_estimators is not None:
            n_estimators_before = self.learner.get_params()["n_estimators"]
            estimators_tmp = self.learner.estimators_[n_estimators:]
            self.learner.estimators_ = self.learner.estimators_[:n_estimators]
            self.learner.set_params(n_estimators=n_estimators)
        proba = self.learner.predict_proba(X)
        if n_estimators is not None:
            self.learner.set_params(n_estimators=n_estimators_before)
            self.learner.estimators_.extend(estimators_tmp)
            assert (len(self.learner.estimators_) == n_estimators_before) and (
                self.learner.get_params()["n_estimators"] == n_estimators_before
            )
        return proba

    def _predict_with_proba_without_transform(self, X, n_estimators=None):
        y_pred_proba = self._predict_proba_after_transform(X, n_estimators=n_estimators)
        y_pred = y_pred_proba.argmax(
            axis=1
        )  # is based on the internal convention that labels are 0 to k-1
        return y_pred, y_pred_proba

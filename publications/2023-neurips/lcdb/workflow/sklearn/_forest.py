import sklearn
from ConfigSpace import (
    Constant,
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
)
from ...experiments.scorer import ClassificationScorer
from ...experiments.utils import get_schedule, filter_keys_with_prefix

import numpy as np

from ._base import SklearnWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.ForestWorkflow",
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
        "max_samples": Float("max_samples", bounds=(0.0, 1.0), default=1.0),
    },
)


class ForestWorkflow(SklearnWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        forest,
        n_estimators,
        timer=None,
        epoch_schedule: str = "full",
        **kwargs,
    ):
        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

        self.max_n_estimators = n_estimators

        super().__init__(
            learner=forest,
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

        # Scoring Schedule for Sub-fidelity
        self.schedule = get_schedule(
            name=epoch_schedule, n=self.max_n_estimators #, base=2, power=0.5, delay=0
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata

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
        y_pred_proba_oob = np.zeros((n_samples, len(np.unique(y))))

        # now grow forest according to the schedule
        for i, n_estimators in enumerate(self.schedule):
            print(n_estimators)
            with self.timer.time("epoch", metadata={"value": n_estimators}):
                with self.timer.time("epoch_train"):
                    self.learner.set_params(n_estimators=n_estimators)
                    self.learner.fit(X, y)

                # compute test scores
                with self.timer.time("epoch_test"):
                    scorer = ClassificationScorer(
                        classes_learner=list(self.learner.classes_),
                        classes_overall=self.infos["classes_overall"],
                        timer=self.timer
                    )
                    with self.timer.time("metrics"):
                        # compute train, validation, and test scores of current forest
                        for label_split, data_split in data.items():
                            with self.timer.time(label_split):
                                with self.timer.time("predict_with_proba"):
                                    (
                                        y_pred,
                                        y_pred_proba,
                                    ) = self._predict_with_proba_without_transform(
                                        data_split["X"]
                                    )

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
                                        y_pred_proba_oob[val_indices] * counters[val_indices]
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
                                assert np.allclose(1, sum_of_probs[mask]), f"NOT A DISTRIBUTION: {y_pred_proba_oob[mask]}"

                                scorer.score(
                                    y_true=y_true[mask],
                                    y_pred=y_pred_oob[mask],
                                    y_pred_proba=y_pred_proba_oob[mask],
                                )

        self.infos["classes"] = list(self.learner.classes_)

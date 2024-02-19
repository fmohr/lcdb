import sklearn
from ConfigSpace import (
    Constant,
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
)
from lcdb.scorer import ClassificationScorer
from sklearn.ensemble import RandomForestClassifier
from lcdb.utils import get_schedule

import numpy as np

from ...utils import filter_keys_with_prefix
from .._preprocessing_workflow import PreprocessedWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.RandomForestWorkflow",
    space={
        "n_estimators": Constant("n_estimators", value=2000),
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


class RandomForestWorkflow(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        random_state=None,
        epoch_schedule: str = "power",
        **kwargs,
    ):
        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="pp@"))
        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

        if max_features == "all":
            max_features = 1.0

        if bootstrap == False:
            max_samples = None

        self.max_n_estimators = n_estimators

        learner_kwargs = dict(
            n_estimators=1,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            warm_start=True,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            random_state=random_state,
        )

        self.learner = RandomForestClassifier(**learner_kwargs)

        # Scoring Schedule for Sub-fidelity
        self.schedule = get_schedule(
            name=epoch_schedule, n=self.max_n_estimators, base=2, power=0.5, delay=0
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata
        X = self.transform(X, y, metadata)
        X_valid = self.transform(X_valid, y_valid, metadata)
        X_test = self.transform(X_test, y_test, metadata)

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
            with self.timer.time("epoch", metadata={"value": n_estimators}):
                with self.timer.time("epoch_train"):
                    self.learner.set_params(n_estimators=n_estimators)
                    self.learner.fit(X, y)

                # compute test scores
                with self.timer.time("epoch_test"):
                    scorer = ClassificationScorer(
                        classes=list(self.learner.classes_), timer=self.timer
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
                            with self.timer.time("oob") as oob_timer:
                                start_index = self.schedule[i - 1] if i > 0 else 0
                                labels = list(self.learner.classes_)
                                for t in range(start_index, n_estimators):
                                    considered_tree = self.learner.estimators_[t]
                                    val_indices = get_unsampled_indices(considered_tree)
                                    y_pred_oob_tree = considered_tree.predict_proba(
                                        X[val_indices]
                                    )
                                    y_pred_proba_oob[val_indices] = (
                                        y_pred_proba_oob[val_indices] * t
                                        + y_pred_oob_tree
                                    ) / (t + 1)

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
                                oob_timer["num_samples"] = num_samples

                                scorer.score(
                                    y_true=y_true[mask],
                                    y_pred=y_pred_oob[mask],
                                    y_pred_proba=y_pred_proba_oob[mask],
                                )

        self.infos["classes"] = list(self.learner.classes_)

    def _predict(self, X):
        X = self.pp_pipeline.transform(X)
        return self.learner.predict(X)

    def _predict_proba(self, X):
        X = self.pp_pipeline.transform(X)
        return self.learner.predict_proba(X)

    def _predict_with_proba_without_transform(self, X):
        y_pred_proba = self.learner.predict_proba(X)
        y_pred = y_pred_proba.argmax(axis=1)
        classes = self.learner.classes_
        y_pred = [classes[i] for i in y_pred]
        return y_pred, y_pred_proba

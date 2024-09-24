import sklearn
from ConfigSpace import (
    Constant,
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
    EqualsCondition,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lcdb.builder.scorer import ClassificationScorer
from lcdb.builder.utils import get_schedule
from lcdb.builder.timer import Timer, Stopwatch

import numpy as np
import time

from ._base import SklearnWorkflow

from tqdm import tqdm

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.TreesEnsembleWorkflow",
    space={
        "n_estimators": Constant("n_estimators", value=512),
        "criterion": Categorical(
            "criterion", items=["gini", "entropy", "log_loss"], default="gini"
        ),
        "max_depth": Integer("max_depth", bounds=(0, 100), default=0),
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
        "splitter": Categorical("splitter", items=["random", "best"], default="best"),
    },
)

CONFIG_SPACE.add_condition(
    EqualsCondition(CONFIG_SPACE["max_samples"], CONFIG_SPACE["bootstrap"], True)
)





class TreesEnsembleWorkflow(SklearnWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        n_estimators=1,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        ccp_alpha=0.0,
        max_samples=None,
        splitter="best",
        random_state=None,
        epoch_schedule: str = "power",
        iterations_to_wait_for_update=4,
        **kwargs,
    ):
        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

        self.max_n_estimators = n_estimators
        self.iterations_to_wait_for_update = iterations_to_wait_for_update
        self.last_recorded_iteration = None

        max_depth = max_depth if max_depth > 0 else None
        max_samples = max_samples if bootstrap else None
        max_features = 1.0 if max_features == "all" else max_features

        if splitter == "best":
            learner = RandomForestClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                random_state=random_state,
            )
        elif splitter == "random":
            learner = ExtraTreesClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                random_state=random_state,
            )
        else:
            raise ValueError(
                f"The splitter is '{splitter}' when it should be in ['random', 'best']."
            )
        super().__init__(learner=learner, timer=timer, **kwargs)

        # Scoring Schedule for Sub-fidelity
        self.schedule = get_schedule(
            name=epoch_schedule, n=self.max_n_estimators
        )
        self.logger.info(f"Initialized tree ensemble with schedule {self.schedule} based on {epoch_schedule}")

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(
        self, X, y, X_valid, y_valid, X_test, y_test, metadata
    ):
        self.metadata = metadata

        t_inner = time.time()  # record inner time to manage time consumption inside this function

        # first train full bagging ensemble. This is because training them iteratively is highly inefficient in sklearn
        self.logger.info(f"Training {self.max_n_estimators} trees.")
        ts_train_start = time.time()
        self.learner.set_params(n_estimators=self.max_n_estimators)
        self.learner.fit(X, y)
        ts_train_stop = time.time()
        total_training_time = ts_train_stop - ts_train_start
        avg_fit_time_per_learner = total_training_time / self.learner.n_estimators
        self.logger.info(f"Trained {self.max_n_estimators} trees in {total_training_time}s.")

        # compute metrics
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

        scorer_timer = Timer()
        scorer_timer.start("artificial_root")
        scorer = ClassificationScorer(
            classes_learner=list(self.learner.classes_),
            classes_overall=self.infos["classes_overall"],
            timer=scorer_timer
        )

        # now compute metrics for partial forest sizes (and simulate the time for training)
        self.logger.info("Computing iteration-wise curve of forest (including out-of-bag fold)")
        y_pred_proba_forest_per_fold = {k: None for k in data.keys()}
        y_pred_proba_forest_oob = np.zeros((n_samples, len(np.unique(y))))
        oob_counters = np.zeros(y_pred_proba_forest_oob.shape)

        stopwatch = Stopwatch()
        stopwatch.start()
        for i, n_estimators in enumerate(tqdm(self.schedule)):

            # determine indices of estimators to be evaluated in this cycle
            indices_of_estimators = list(range(self.schedule[i - 1] if i > 0 else 0, self.schedule[i]))

            t_epoch_start = time.time()
            self.logger.debug(f"Forest Size {n_estimators}")

            self.timer.start("epoch", metadata={"value": n_estimators}, timestamp_start=t_inner)

            # simulate training time
            self.timer.start("epoch_train", timestamp_start=t_inner)
            t_inner += avg_fit_time_per_learner
            self.timer.stop(timestamp_end=t_inner)

            # compute test scores
            t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
            self.timer.start("epoch_test", timestamp_start=t_inner)
            t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
            self.timer.start("metrics", timestamp_start=t_inner)

            # compute train, validation, and test scores of current forest
            for label_split, data_split in data.items():

                y_pred_proba_forest = y_pred_proba_forest_per_fold[label_split]

                t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
                self.timer.start(label_split, timestamp_start=t_inner)
                t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
                self.timer.start("predict_with_proba", timestamp_start=t_inner)

                for i_estimator in indices_of_estimators:

                    # update forest prediction
                    y_pred, y_pred_proba = self._predict_with_proba_of_single_tree_without_transform(
                        X=data_split["X"],
                        i_estimator=i_estimator
                    )
                    assert len(data_split["X"]) == len(y_pred_proba)
                    if y_pred_proba_forest is None:
                        y_pred_proba_forest = y_pred_proba
                    else:
                        y_pred_proba_forest += 1 / (i_estimator + 1) * (y_pred_proba - y_pred_proba_forest)
                    y_pred_forest = y_pred_proba_forest.argmax(axis=1)
                    self.logger.debug(
                        f"Updated probabilistic forest predictions for fold {label_split} at size {i_estimator + 1}."
                    )
                t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
                self.timer.stop(timestamp_end=t_inner)  # exit from predict_proba

                # compute score based on current prediction (but only at check points)
                y_true = data_split["y"]
                with scorer_timer.time(label_split):
                    scorer.score(
                        y_true=y_true,
                        y_pred=y_pred_forest,
                        y_pred_proba=y_pred_proba_forest,
                    )

                # attach scores to current timer
                offset = t_inner - scorer_timer.active_node.children[-1].timestamp_start
                self.timer.inject(scorer_timer.active_node.children[-1], ignore_root=True, offset=offset)
                stopwatch.checkpoint().elapsed_time_since_last_checkpoint
                t_inner = self.timer.active_node.children[-1].timestamp_end
                self.timer.stop(timestamp_end=t_inner)  # exit from fold

            """
                maybe add OOB fold (in case of bootstrapping);
                this requires again predictions from all trees on a sub-set of training data but is necessary
                for clean separation between training and OOB prediction times
            """
            if self.learner.bootstrap:

                # compute train, validation, and test scores of current forest
                t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
                self.timer.start("oob", timestamp_start=t_inner)
                labels = list(self.learner.classes_)
                n_labels = len(labels)

                # update OOB prediction of forest
                for t in indices_of_estimators:
                    considered_tree = self.learner.estimators_[t]
                    val_indices = get_unsampled_indices(considered_tree)
                    y_pred_oob_tree = considered_tree.predict_proba(X[val_indices])
                    y_pred_proba_forest_oob[val_indices] = (
                        y_pred_proba_forest_oob[val_indices]
                        * oob_counters[val_indices]
                        + y_pred_oob_tree
                    ) / (oob_counters[val_indices] + 1)
                    oob_counters[val_indices] += 1
                y_pred_forest_oob = np.array(
                    [
                        labels[ind]
                        for ind in np.argmax(y_pred_proba_forest_oob, axis=1)
                    ]
                )
                y_true = y

                # Only keep OOB samples
                sum_of_probs = y_pred_proba_forest_oob.sum(axis=1)
                mask = sum_of_probs > 0.0
                num_samples = np.sum(mask)
                self.timer.active_node["num_samples"] = int(num_samples)
                assert np.allclose(
                    1, sum_of_probs[mask]
                ), f"NOT A DISTRIBUTION: {y_pred_proba_forest[mask]}"

                with scorer_timer.time("oob"):
                    scorer.score(
                        y_true=y_true[mask],
                        y_pred=y_pred_forest_oob[mask],
                        y_pred_proba=y_pred_proba_forest_oob[mask],
                    )

                # attach scores to current timer
                offset = t_inner - scorer_timer.active_node.children[-1].timestamp_start
                self.timer.inject(scorer_timer.active_node.children[-1], ignore_root=True, offset=offset)
                stopwatch.checkpoint().elapsed_time_since_last_checkpoint
                t_inner = self.timer.active_node.children[-1].timestamp_end
                self.timer.stop(timestamp_end=t_inner)  # leave from OOB

            t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
            self.timer.stop(timestamp_end=t_inner) # leave from metrics
            t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
            self.timer.stop(timestamp_end=t_inner) # leave from epoch test
            t_inner += stopwatch.checkpoint().elapsed_time_since_last_checkpoint
            self.timer.stop(timestamp_end=t_inner)  # leave from epoch
            t_epoch_end = time.time()
            self.logger.debug(f"Finished epoch {n_estimators} within {round(t_epoch_end - t_epoch_start, 4)}s")

        scorer_timer.stop()
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

    def _predict_with_proba_of_single_tree_without_transform(self, X, i_estimator):
        y_pred_proba = self.learner[i_estimator].predict_proba(X)
        y_pred = y_pred_proba.argmax(
            axis=1
        )  # is based on the internal convention that labels are 0 to k-1
        return y_pred, y_pred_proba

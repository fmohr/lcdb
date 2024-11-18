import numpy as np
from ConfigSpace import ConfigurationSpace, Float, Integer, Uniform
from lcdb.builder.scorer import ClassificationScorer
from lcdb.builder.utils import filter_keys_with_prefix, get_schedule
from .._preprocessing_workflow import PreprocessedWorkflow
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix, XGBClassifier
from xgboost.callback import TrainingCallback


CONFIG_SPACE = ConfigurationSpace(
    name="xgboost",
    space={
        "learning_rate": Float(
            "learning_rate",
            bounds=(10**-6, 1),
            distribution=Uniform(),
            default=0.3,
            log=True,
        ),
        "gamma": Float(
            "gamma",
            bounds=(10**-6, 2**6),
            distribution=Uniform(),
            default=10**-6,
            log=True,
        ),  # normal default would be 0
        "min_child_weight": Float(
            "min_child_weight",
            bounds=(10**-6, 2**5),
            distribution=Uniform(),
            default=1,
            log=True,
        ),
        "max_depth": Integer(
            "max_depth", bounds=(2, 2**5), distribution=Uniform(), default=6, log=True
        ),
        "subsample": Float(
            "subsample", bounds=(0.5, 1), distribution=Uniform(), default=1
        ),
        "colsample_bytree": Float(
            "colsample_bytree", bounds=(0.3, 1), distribution=Uniform(), default=1
        ),
        "reg_alpha": Float(
            "reg_alpha",
            bounds=(10**-6, 2),
            distribution=Uniform(),
            default=10**-6,
            log=True,
        ),  # normal default would be 0
        "reg_lambda": Float(
            "reg_lambda",
            bounds=(10**-6, 2),
            distribution=Uniform(),
            default=1,
            log=True,
        ),
    },
)


class EvalCallBack(TrainingCallback):
    def __init__(self, workflow, timer, encoder, data, schedule: str = "power"):
        super().__init__()
        self.timer = timer
        self.encoder = encoder
        self.workflow = workflow
        self.scorer = None
        self.data = data
        self.schedule = schedule

        self.epoch = None
        self.train_timer_id = None
        self.test_timer_id = None
        self.epoch_timer_id = None

    def before_iteration(self, model, epoch, evals_log):
        # start tracking time for the current anchor (epoch)
        self.epoch = epoch
        self.epoch_timer_id = self.timer.start("epoch", metadata={"value": self.epoch})
        # start tracking time for the training
        self.train_timer_id = self.timer.start("epoch_train")
        return False

    def after_iteration(self, model, epoch, evals_log):
        assert self.timer.active_node.id == self.train_timer_id
        self.timer.stop()
        if (epoch + 1) in self.schedule:
            self.workflow.logger.info(f"Computing iteration-wise performance curve at iteration anchor {epoch + 1}")
            self.test_timer_id = self.timer.start("epoch_test")
            with self.timer.time("metrics"):
                for label_split, data_split in self.data.items():
                    with self.timer.time(label_split):
                        with self.timer.time("predict_with_proba"):
                            y_pred_proba = model.predict(data_split["X"], strict_shape=True)

                        y_pred_proba = self.workflow._create_full_probs(y_pred_proba)
                        y_pred = self.workflow._predict_label_from_probs(y_pred_proba, orig_label=False)

                        y_true = data_split["y"]

                        # get the scorer lazy, because at init time, the labels are not yet registered
                        if self.scorer is None:
                            self.scorer = ClassificationScorer(
                                classes_learner=self.workflow.infos["classes_train"],
                                classes_overall=self.workflow.infos["classes_overall"],
                                timer=self.timer
                            )
                        out = self.scorer.score(
                            y_true=y_true,
                            y_pred=y_pred,
                            y_pred_proba=y_pred_proba,
                        )
                        self.workflow.logger.debug(f"{label_split} accuracy: {np.diag(out['confusion_matrix']).sum() / np.sum(out['confusion_matrix'])}")
            assert self.timer.active_node.id == self.test_timer_id
            self.timer.stop()
            assert self.timer.active_node.id == self.epoch_timer_id
        self.timer.stop()
        return False


class XGBoostWorkflow(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    # FIXME: increase the number of iterations to something like 1000-10000
    # FIXME: trycatch and logging?
    def __init__(
        self,
        timer=None,
        n_estimators=2048,
        learning_rate=0.3,
        gamma=10**-6,
        min_child_weight=0,
        max_depth=6,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=10**-6,
        reg_lambda=1,
        random_state=None,
        logger=None,
        epoch_schedule="power",
        raise_exception_on_unsuitable_preprocessor=True,
        **kwargs,
    ):
        super().__init__(
            timer=timer,
            logger=logger,
            random_state=random_state,
            raise_exception_on_unsuitable_preprocessor=raise_exception_on_unsuitable_preprocessor,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True
        self.n_estimators = n_estimators

        learner_kwargs = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )

        self.learner = XGBClassifier(**learner_kwargs)

        self.schedule = get_schedule(
            name=epoch_schedule, n=self.n_estimators
        )

        self.encoder = LabelEncoder()
        self.labels_missing_in_train_set = None  # will be set in fit
        self.n_classes = None

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return True

    @classmethod
    def is_randomizable(cls):
        return True

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        # FIXME: not sure what is the best way to additionally track the time spent in the fit method
        #   we could try to also track some of the overhead here for example the y transform but this is not real
        #   like in outer workflow

        # store the metadata and label encode y, y_valid and y_test
        self.metadata = metadata
        self.encoder.fit(y)
        self.labels_missing_in_train_set = sorted(set(self.infos["classes_overall"]).difference(set(self.infos["classes_train"])))
        self.n_classes = len(self.infos["classes_train"])

        # construct callback that will handle the iteration-wise learning curve tracking and set it as a callback
        eval_callback = EvalCallBack(
            workflow=self,
            timer=self.timer,
            encoder=self.encoder,
            data=dict(train=dict(X=DMatrix(X, label=y), y=y), val=dict(X=DMatrix(X_valid), y=y_valid), test=dict(X=DMatrix(X_test), y=y_test)),
            schedule=self.schedule
        )
        self.learner.set_params(callbacks=[eval_callback])

        multiclass = self.n_classes > 2

        if multiclass:
            self.learner.set_params(objective="multi:softprob")
        else:
            self.learner.set_params(objective="binary:logistic")

        # fit the learner
        self.logger.info(f"Starting training of XGBoost with schedule {self.schedule}")
        self.learner.fit(X, self.encoder.transform(y))
        self.logger.info("Training of XGBoost finished")

    def _create_full_probs(self, probs_pred):
        if self.n_classes == 2 and probs_pred.shape[1] == 1:
            # add the first class (0) to the probabilities
            probs_pred = np.concatenate((1 - probs_pred, probs_pred), axis=1)
        probs_pred /= probs_pred.sum(axis=1, keepdims=True)  # make sure that probs sum to 1
        return probs_pred

    def _predict_label_from_probs(self, probs, orig_label=True):
        key = "classes_train"
        if orig_label:
            key += "_orig"
        return np.array([
            self.infos[key][i] for i in np.argmax(probs, axis=1)
        ])

    def _predict_proba_after_transform(self, X):
        y_prob = self.learner.predict_proba(X)
        assert y_prob.shape[1] <= self.n_classes, "XGBoost has created a distribution with more columns than there are classes."
        out = self._create_full_probs(y_prob)
        assert out.shape[1] <= self.n_classes, "_create_full_probs has created a distribution with more columns than there are classes."
        return out

    def _predict_after_transform(self, X, orig_label=True):
        return self._predict_label_from_probs(self._predict_proba_after_transform(X), orig_label=orig_label)


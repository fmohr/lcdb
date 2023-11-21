import numpy as np
from ConfigSpace import ConfigurationSpace, Float, Integer, Uniform
from lcdb.scorer import ClassificationScorer
#from lcdb.utils import filter_keys_with_prefix
#from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from ...utils import filter_keys_with_prefix
from .._preprocessing_workflow import PreprocessedWorkflow
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix, XGBClassifier
from xgboost.callback import TrainingCallback


class ExtendedLabelEncoder(LabelEncoder):
    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown
        self.unknown_integer = None
        LabelEncoder.__init__(self)

    def fit(self, y):
        # unknown integer is n if n is the number of seen classes
        # because we map seen class labels to (0 to n-1) and n is the next unseen class
        if self.handle_unknown == "ignore":
            if "___UNKNOWN___" in y:
                raise ValueError("___UNKNOWN___ is a reserved label")
        self.unknown_integer = len(np.unique(y))
        return super(ExtendedLabelEncoder, self).fit(y)

    def transform(self, y):
        if self.handle_unknown == "ignore":
            return [super(ExtendedLabelEncoder, self).transform([label])[
                        0] if label in self.classes_ else self.unknown_integer for label in y]
        else:
            return super(ExtendedLabelEncoder, self).transform(y)

    def inverse_transform(self, y):
        if self.handle_unknown == "ignore":
            return [super(ExtendedLabelEncoder, self).inverse_transform([label])[0] if label != self.unknown_integer else "___UNKNOWN___" for label in y]
        else:
            return super(ExtendedLabelEncoder, self).inverse_transform(y)


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
    def __init__(self, workflow, timer, encoder, data):
        super().__init__()
        self.timer = timer
        self.workflow = workflow
        self.scorer = ClassificationScorer(classes=self.workflow.infos["classes"], timer=self.timer)
        self.encoder = encoder
        self.data = data
        self.n_classes = len(self.workflow.infos["classes"])

        self.epoch = None
        self.train_timer_id = None
        self.test_timer_id = None
        self.epoch_timer_id = None

        self.data["train"]["y"] = self.encoder.inverse_transform(self.data["train"]["y"])
        self.data["val"]["y"] = self.encoder.inverse_transform(self.data["val"]["y"])
        self.data["test"]["y"] = self.encoder.inverse_transform(self.data["test"]["y"])

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
        self.test_timer_id = self.timer.start("epoch_test")
        with self.timer.time("metrics"):
            for label_split, data_split in self.data.items():
                with self.timer.time(label_split):
                    with self.timer.time("predict_with_proba"):
                        y_pred_proba = model.predict(data_split["X"], strict_shape=True)

                    y_pred_proba = self.create_full_probs(y_pred_proba)
                    y_pred = self.create_labels_from_probs(y_pred_proba, invert=True)

                    y_true = data_split["y"]

                    self.scorer.score(
                        y_true=y_true,
                        y_pred=y_pred,
                        y_pred_proba=y_pred_proba,
                    )
        assert self.timer.active_node.id == self.test_timer_id
        self.timer.stop()
        assert self.timer.active_node.id == self.epoch_timer_id
        self.timer.stop()
        return False

    #def before_training(self, model):
    #    # start tracking time for the training
    #    self.train_timer_id = self.timer.start("epoch_train")
    #    return model

    #def after_training(self, model):
    #    assert self.timer.active_node.id == self.train_timer_id
    #    self.timer.stop()
    #    self.test_timer_id = self.timer.start("epoch_test")
    #    with self.timer.time("metrics"):
    #        for label_split, data_split in self.data.items():
    #            with self.timer.time(label_split):
    #                with self.timer.time("predict_with_proba"):
    #                    y_pred_proba = model.predict(data_split["X"], strict_shape=True)

    #                y_pred_proba = self.create_full_probs(y_pred_proba)
    #                y_pred = self.create_labels_from_probs(y_pred_proba, invert=True)

    #                y_true = data_split["y"]

    #                self.scorer.score(
    #                    y_true=y_true,
    #                    y_pred=y_pred,
    #                    y_pred_proba=y_pred_proba,
    #                )
    #    self.timer.stop()
    #    return model

    def create_full_probs(self, probs_pred):
        if self.n_classes == 2:
            # add the first class (0) to the probabilities
            probs_pred = np.concatenate((1 - probs_pred, probs_pred), axis=1)
        return probs_pred

    def create_labels_from_probs(self, probs_pred, invert: bool = False):
        # get the index of the highest probability
        # this is the same procedure the official sklearn wrapper uses
        # https://github.com/dmlc/xgboost/blob/6c0a190f6d12d2ba6a1cabd7741881ea1913d433/python-package/xgboost/sklearn.py#L1568
        y_pred = np.argmax(probs_pred, axis=1)
        # invert the label encoding index if necessary
        if invert:
            y_pred = self.encoder.inverse_transform(y_pred)
        return y_pred


class XGBoostWorkflow(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    # FIXME: drop some unnecessary preprocessing steps
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    # FIXME: increase the number of iterations to something like 1000-10000
    # FIXME: random_state?
    # FIXME: trycatch and logging?
    def __init__(
        self,
        timer=None,
        n_estimators=100,
        learning_rate=0.3,
        gamma=10**-6,
        min_child_weight=0,
        max_depth=6,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=10**-6,
        reg_lambda=1,
        random_state=None,
        **kwargs,
    ):
        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="pp@"))

        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

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

        self.encoder = ExtendedLabelEncoder()

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        # FIXME: not sure what is the best way to additionally track the time spent in the fit method
        #   we could try to also track some of the overhead here for example the y transform but this is not real
        #   like in outer workflow

        # store the metadata and label encode y, y_valid and y_test
        self.metadata = metadata
        self.encoder.fit(y)
        self.infos["classes"] = list(self.encoder.classes_)
        y = self.encoder.transform(y)
        y_valid = self.encoder.transform(y_valid)
        y_test = self.encoder.transform(y_test)

        # transform X, X_valid and X_test
        X = self.transform(X, y, metadata)
        X_valid = self.transform(X_valid, y_valid, metadata)
        X_test = self.transform(X_test, y_test, metadata)

        # construct callback that will handle the iteration-wise learning curve tracking and set it as a callback
        eval_callback = EvalCallBack(
            workflow=self,
            timer=self.timer,
            encoder=self.encoder,
            data=dict(train=dict(X=DMatrix(X, label=y), y=y), val=dict(X=DMatrix(X_valid), y=y_valid), test=dict(X=DMatrix(X_test), y=y_test)),
        )
        self.learner.set_params(callbacks=[eval_callback])

        n_classes = len(self.infos["classes"])
        multiclass = n_classes > 2

        if multiclass:
            self.learner.set_params(objective="multi:softprob")
        else:
            self.learner.set_params(objective="binary:logistic")

        # fit the learner
        self.learner.fit(X, y)

    def _predict(self, X):
        X_pred = self.pp_pipeline.transform(X)
        y_pred = self.learner.predict(X_pred)
        return self.encoder.inverse_transform(y_pred)

    def _predict_proba(self, X):
        X_pred = self.pp_pipeline.transform(X)
        y_pred = self.learner.predict_proba(X_pred)
        return y_pred

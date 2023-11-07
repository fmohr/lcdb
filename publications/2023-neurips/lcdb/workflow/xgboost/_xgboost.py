import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace, Float, Integer, Uniform
from lcdb.curve import Curve
from lcdb.curvedb import CurveDB
from lcdb.timer import Timer
from lcdb.utils import filter_keys_with_prefix
from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
# from ...utils import filter_keys_with_prefix
# from .._preprocessing_workflow import PreprocessedWorkflow
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


# FIXME: double check the timings
class EvalCallBack(TrainingCallback):
    def __init__(self, data_train, data_valid, data_test, workflow):
        super().__init__()
        self.timer = workflow.fidelities_timer
        self.cur_anchor = None
        self.labels_as_used_by_workflow = None
        self.workflow = workflow
        self.encoder = workflow.encoder
        self.curves = workflow.fidelities
        self.DM_train = DMatrix(data_train[0], label=data_train[1])
        self.y_train = data_train[1]
        self.DM_valid = DMatrix(data_valid[0])
        self.y_valid = data_valid[1]
        self.DM_test = DMatrix(data_test[0])
        self.y_test = data_test[1]
        self.n_classes = len(self.encoder.classes_)

        self.y_train = self.encoder.inverse_transform(self.y_train)
        self.y_valid = self.encoder.inverse_transform(self.y_valid)
        self.y_test = self.encoder.inverse_transform(self.y_test)

    def before_iteration(self, model, epoch, evals_log):
        # start tracking time for the current anchor (epoch)
        self.timer.enter(epoch)
        self.timer.start("fit")
        return False

    def after_iteration(self, model, epoch, evals_log):
        self.timer.stop("fit")
        # fit is done so we stop tracking the fit time for the current anchor (epoch)
        # and compute the metrics for the current anchor (epoch)
        self.cur_anchor = epoch
        self.compute_metrics_for_workflow(model)
        self.timer.leave()
        return False

    def create_full_probs(self, probs_pred, n_classes):
        if n_classes == 2:
            # add the first class (0) to the probabilities
            probs_pred = np.concatenate((1 - probs_pred, probs_pred), axis=1)
        return probs_pred

    def create_labels_from_probs(self, probs_pred, invert: bool = False):
        # get the index of the highest probability
        y_pred = np.argmax(probs_pred, axis=1)
        # invert the label encoding index if necessary
        if invert:
            y_pred = self.encoder.inverse_transform(y_pred)
        return y_pred

    def get_predictions(self, model):
        # here we also need to track the time spent in the predict and predict_proba methods
        # this is somewhat hacky because the xgboost model only requires a single predict proba call
        # but we track it separately nonetheless (predict proba to full and argmax to labels with invert)
        keys = {}
        labels = self.encoder.classes_

        for X_, y_true, postfix in [
            (self.DM_train, self.y_train, "train"),
            (self.DM_valid, self.y_valid, "val"),
            (self.DM_test, self.y_test, "test"),
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.timer.enter(postfix)
                self.timer.start("predict_proba")
                y_pred_proba = model.predict(X_, strict_shape=True)
                y_pred_proba = self.create_full_probs(y_pred_proba, self.n_classes)
                self.timer.stop("predict_proba")
                self.timer.start("predict")
                y_pred = self.create_labels_from_probs(y_pred_proba, invert=True)
                self.timer.stop("predict")
                keys[f"y_pred_{postfix}"] = y_pred
                keys[f"y_pred_proba_{postfix}"] = y_pred_proba
                self.timer.leave()
        return keys, labels

    def compute_metrics_for_workflow(self, model):
        predictions, labels = self.get_predictions(model)
        self.labels_as_used_by_workflow = labels
        return self.extend_curves_based_on_predictions(**predictions)

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
            (self.y_train, y_pred_train, y_pred_proba_train, "train"),
            (self.y_valid, y_pred_val, y_pred_proba_val, "val"),
            (self.y_test, y_pred_test, y_pred_proba_test, "test"),
        ]:
            self.timer.enter(postfix)
            curve = self.curves[postfix]
            curve.compute_metrics(self.cur_anchor, y_true, y_pred, y_pred_proba)
            self.timer.leave()


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
        super().__init__(**filter_keys_with_prefix(kwargs, prefix="pp@"))

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

        self.encoder = LabelEncoder()

        self.fidelities_timer = None
        self.fidelities = None
        self.fit_report = None

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        # FIXME: not sure what is the best way to additionally track the time spent in the fit method
        # we could try to also track some of the overhead here for example the y transform but this is not real
        # like in outer workflow

        # set up fidelities timer and fidelity curves and fit report
        self.fidelities_timer = Timer(precision=6)
        self.fidelities = {
            "train": Curve(workflow=self, timer=self.fidelities_timer),
            "val": Curve(workflow=self, timer=self.fidelities_timer),
            "test": Curve(workflow=self, timer=self.fidelities_timer),
        }
        self.fit_report = {}

        # store the metadata and label encode y, y_valid and y_test
        self.metadata = metadata
        y = self.encoder.fit_transform(y)
        self.infos["classes"] = list(self.encoder.classes_)
        y_valid = self.encoder.transform(y_valid)
        y_test = self.encoder.transform(y_test)

        # transform X, X_valid and X_test
        X = self.transform(X, y, metadata)
        X_valid = self.transform(X_valid, y_valid, metadata)
        X_test = self.transform(X_test, y_test, metadata)

        # construct callback that will handle the fidelity curves and the fit report and set it as callback
        eval_callback = EvalCallBack(
            (X, y),
            data_valid=(X_valid, y_valid),
            data_test=(X_test, y_test),
            workflow=self,
        )
        self.learner.set_params(callbacks=[eval_callback])

        # fit the learner
        self.learner.fit(X, y)

        # store the fit report
        self.fit_report["fidelities_db"] = CurveDB(
            self.fidelities["train"],
            self.fidelities["val"],
            self.fidelities["test"],
            self.fidelities_timer.runtimes,
            None,
        ).dump_to_dict()

    def _predict(self, X):
        X_pred = self.pp_pipeline.transform(X)
        y_pred = self.learner.predict(X_pred)
        return self.encoder.inverse_transform(y_pred)

    def _predict_proba(self, X):
        X_pred = self.pp_pipeline.transform(X)
        y_pred = self.learner.predict_proba(X_pred)
        return y_pred

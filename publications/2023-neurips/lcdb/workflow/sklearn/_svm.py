from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC

from lcdb.builder.utils import decision_fun_to_proba
from ._base import SklearnWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.SVMWorkflow",
    space={
        "multi_class": Categorical("multi_class", ["ovr", "ovo-scikit"], default="ovr"),
    }
)


class SVMWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        svm_instance,
        timer=None,
        multi_class="ovr",
        **kwargs,
    ):

        if multi_class == "ovr":
            learner = svm_instance
        else:
            learner = OneVsOneClassifier(svm_instance, n_jobs=None)

        super().__init__(learner, timer, **kwargs)

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return False

    @classmethod
    def is_randomizable(cls):
        return True

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        super()._fit_model_after_transformation(X, y, X_valid, y_valid, X_test, y_test, metadata)
        if type(self.learner) is LinearSVC or type(self.learner) is SVC:
            self.infos["n_iter_"] = self.learner.n_iter_
        else:
            n_iter_ = []
            for est in self.learner.estimators_:
                n_iter_.append(est.n_iter_)
            self.infos["n_iter_"] = n_iter_

    def _predict_proba_after_transform(self, X):
        return decision_fun_to_proba(self.learner.decision_function(X))

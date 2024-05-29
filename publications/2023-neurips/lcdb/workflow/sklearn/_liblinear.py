from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    Integer,
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
import numpy as np
from scipy.special import softmax

from ...utils import filter_keys_with_prefix
from .._preprocessing_workflow import PreprocessedWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.LibLinearWorkflow",
    space={
        "dual": Categorical("dual", [False, True], default=True),
        "C": Float("C", bounds=(1e-12, 1e12), default=1, log=True),
        "multi_class": Categorical("multiclass", ["ovr", "ovo-scikit"], default="ovr"),
        "tol": Float("tol", bounds=(4.5e-5, 2), default=1e-3, log=True),
        "max_iter": Integer("max_iter", bounds=(100, 10000), default=1000, log=True),
        "class_weight": Categorical(
            "class_weight", items=["balanced", "none"], default="none"
        ),
        "loss": Categorical(
            "loss", ["hinge", "squared_hinge"], default="squared_hinge"
        ),
        "penalty": Categorical("penalty", ["l2", "l1"], default="l2"),
        "fit_intercept": Categorical("fit_intercept", [False, True], default=True),
        "intercept_scaling": Float(
            "intercept_scaling", bounds=(1.0, 1e3), default=1.0, log=True
        ),
    },
)

forbidden_clause_a = ForbiddenEqualsClause(CONFIG_SPACE["loss"], "hinge")
forbidden_clause_b = ForbiddenEqualsClause(CONFIG_SPACE["penalty"], "l1")

forbidden_clause = ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_b)
CONFIG_SPACE.add_forbidden_clause(forbidden_clause)

forbidden_clause_c = ForbiddenEqualsClause(CONFIG_SPACE["loss"], "squared_hinge")
forbidden_clause_d = ForbiddenEqualsClause(CONFIG_SPACE["penalty"], "l1")
forbidden_clause_e = ForbiddenEqualsClause(CONFIG_SPACE["dual"], True)

forbidden_clause2 = ForbiddenAndConjunction(forbidden_clause_c, forbidden_clause_d)
forbidden_clause3 = ForbiddenAndConjunction(forbidden_clause2, forbidden_clause_e)
CONFIG_SPACE.add_forbidden_clause(forbidden_clause3)

forbidden_clause_f = ForbiddenEqualsClause(CONFIG_SPACE["loss"], "hinge")
forbidden_clause_g = ForbiddenEqualsClause(CONFIG_SPACE["penalty"], "l2")
forbidden_clause_h = ForbiddenEqualsClause(CONFIG_SPACE["dual"], False)

forbidden_clause4 = ForbiddenAndConjunction(forbidden_clause_f, forbidden_clause_g)
forbidden_clause5 = ForbiddenAndConjunction(forbidden_clause4, forbidden_clause_h)
CONFIG_SPACE.add_forbidden_clause(forbidden_clause5)


class LibLinearWorkflow(PreprocessedWorkflow):
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
        dual=True,
        C=1,
        multi_class="ovr",
        tol=1e-3,
        max_iter=1000,
        class_weight="none",
        loss="squared_hinge",
        penalty="l2",
        fit_intercept=True,
        intercept_scaling=1.0,
        random_state=None,
        **kwargs,
    ):
        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="pp@"))

        learner_kwargs = dict(
            dual=dual,
            C=C,
            tol=tol,
            multi_class="ovr",
            max_iter=max_iter,
            class_weight=None if class_weight == "none" else class_weight,
            loss=loss,
            penalty=penalty,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            random_state=random_state,
        )

        if multi_class == "ovr":
            self.learner = LinearSVC(**learner_kwargs)
        else:
            self.learner = OneVsOneClassifier(LinearSVC(**learner_kwargs), n_jobs=None)

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata
        self.learner.fit(X, y)

        self.infos["classes"] = list(self.learner.classes_)
        if type(self.learner) is LinearSVC:
            self.infos["n_iter_"] = self.learner.n_iter_
        else:
            n_iter_ = []
            for est in self.learner.estimators_:
                n_iter_.append(est.n_iter_)
            self.infos["n_iter_"] = n_iter_

    def _predict_after_transform(self, X):
        return self.learner.predict(X)

    def _predict_proba_after_transform(self, X):
        decision_fun_vals = self.learner.decision_function(X)
        sigmoid = lambda z: 1/(1 + np.exp(-z))
        if len(decision_fun_vals.shape) == 2:
            return softmax(decision_fun_vals, axis=1)
        else:
            a = sigmoid(decision_fun_vals)
            return np.column_stack([a, 1 - a])

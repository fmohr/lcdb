# Define the Config Space
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    Integer,
    OrConjunction,
    Uniform,
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

from .._base_workflow import BaseWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="liblinear",
    space={
        "dual": Categorical("dual", [False, True], default=True),
        "C": Float(
            "C", bounds=(1e-12, 1e12), distribution=Uniform(), default=1, log=True
        ),
        "multiclass": Categorical("multiclass", ["ovr", "ovo-scikit"], default="ovr"),
        "tol": Float(
            "tol", bounds=(4.5e-5, 2), distribution=Uniform(), default=1e-3, log=True
        ),
        # "max_iter": Constant("max_iter", 1000),
        "max_iter": Integer("max_iter", bounds=(100, 10000), log=True),
        "class_weight": Categorical("class_weight", ["balanced", "none"]),
        "loss": Categorical(
            "loss", ["hinge", "squared_hinge"], default="squared_hinge"
        ),
        "penalty": Categorical("penalty", ["l2", "l1"], default="l2"),
        "scaler": Categorical(
            "scaler", ["minmax", "standardize", "none"], default="none"
        ),
    },
)

fit_intercept = Categorical("fit_intercept", [False, True], default=True)
intercept_scaling = Float("intercept_scaling", bounds=(1.0, 1e3), default=1.0, log=True)
CONFIG_SPACE.add_hyperparameters([fit_intercept, intercept_scaling])

#! Commented out because it can create an error
# sklearn.utils._param_validation.InvalidParameterError: The 'intercept_scaling' parameter of LinearSVC must be a float in the range (0.0, inf). Got nan instead.
# cond_2 = EqualsCondition(intercept_scaling, fit_intercept, True)
# CONFIG_SPACE.add_conditions([cond_2])

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


class LibLinearWorkflow(BaseWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(self, **kwargs):
        super().__init__()

        hyperparams = kwargs.copy()
        hyperparams["verbose"] = False
        hyperparams.pop("scaler")

        if hyperparams["class_weight"] == "none":
            hyperparams["class_weight"] = None

        if hyperparams["multiclass"] == "ovr":
            del hyperparams["multiclass"]
            hyperparams["multi_class"] = "ovr"
            self.learner = LinearSVC(**hyperparams)
            return
        if hyperparams["multiclass"] == "ovo-scikit":
            del hyperparams["multiclass"]
            hyperparams["multi_class"] = "ovr"
            self.learner = OneVsOneClassifier(LinearSVC(**hyperparams), n_jobs=None)
            return

        raise Exception("Multiclass strategy not implemented")

    def update_summary(self):
        pass

    @classmethod
    def config_space(cls):
        # TODO: If the config_space needs to be expanded with preprocessing module it should be done here
        return cls._config_space

    def _fit(self, X, y):
        self.learner.fit(X, y)

        if type(self.learner) is LinearSVC:
            self.infos["n_iter_"] = self.learner.n_iter_
        else:
            n_iter_ = []
            for est in self.learner.estimators_:
                n_iter_.append(est.n_iter_)
            self.infos["n_iter_"] = n_iter_

    def predict_proba(self, X):
        raise Exception("Sorry, we don't do probabilistic classification with SVMs")

    def _predict(self, X):
        return self.learner.predict(X)

    def decision_function(self, X):
        return self.learner.decision_function(X)

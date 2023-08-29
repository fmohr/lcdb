# Define the Config Space
import numpy as np
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    Integer,
    Uniform,
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
    OrdinalEncoder,
)
from sklearn.svm import LinearSVC

from .._base_workflow import BaseWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn._liblinear",
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
        # TODO: refine preprocessing
        "transform_real": Categorical(
            "transform_real", ["minmax", "std", "none"], default="none"
        ),
        "transform_cat": Categorical(
            "transform_cat", ["onehot", "ordinal"], default="onehot"
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


class LibLinearWorkflow(BaseWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(
        self,
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
        transform_real="none",
        transform_cat="onehot",
        **kwargs,
    ):
        super().__init__()

        self.transform_real = transform_real
        self.transform_cat = transform_cat

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
        )

        if multi_class == "ovr":
            self.learner = LinearSVC(**learner_kwargs)
        else:
            self.learner = OneVsOneClassifier(LinearSVC(**learner_kwargs), n_jobs=None)

    @classmethod
    def config_space(cls):
        # TODO: If the config_space needs to be expanded with preprocessing module it should be done here
        return cls._config_space

    def _transform(self, X, metadata):
        X_cat = X[:, metadata["categories"]["columns"]]
        X_real = X[:, ~metadata["categories"]["columns"]]

        has_cat = X_cat.shape[1] > 0
        has_real = X_real.shape[1] > 0

        if not (self.transform_fitted):
            # Categorical features
            if self.transform_cat == "onehot":
                self._transformer_cat = OneHotEncoder(drop="first", sparse_output=False)
            elif self.transform_cat == "ordinal":
                self._transformer_cat = OrdinalEncoder()
            else:
                raise ValueError(
                    f"Unknown categorical transformation {self.transform_cat}"
                )

            if metadata["categories"]["values"] is not None:
                max_categories = max(len(x) for x in metadata["categories"]["values"])
                values = np.array(
                    [
                        c_val + [c_val[-1]] * (max_categories - len(c_val))
                        for c_val in metadata["categories"]["values"]
                    ]
                ).T
            else:
                values = X_cat

            if has_cat:
                self._transformer_cat.fit(values)

            # Real features
            if self.transform_real == "minmax":
                self._transformer_real = MinMaxScaler()
            elif self.transform_real == "std":
                self._transformer_real = StandardScaler()
            elif self.transform_real == "none":
                # No transformation
                self._transformer_real = FunctionTransformer(func=lambda x: x)
            else:
                raise ValueError(f"Unknown real transformation {self.transform_real}")

            if has_real:
                self._transformer_real.fit(X_real)

        if has_cat:
            X_cat = self._transformer_cat.transform(X_cat)
        if has_real:
            X_real = self._transformer_real.transform(X_real)
        X = np.concatenate([X_real, X_cat], axis=1)
        return X

    def _fit(self, X, y, metadata):
        self.metadata = metadata
        X = self.transform(X, metadata)

        self.learner.fit(X, y)

        if type(self.learner) is LinearSVC:
            self.infos["n_iter_"] = self.learner.n_iter_
        else:
            n_iter_ = []
            for est in self.learner.estimators_:
                n_iter_.append(est.n_iter_)
            self.infos["n_iter_"] = n_iter_

    def _predict(self, X):
        X = self.transform(X, metadata=self.metadata)
        return self.learner.predict(X)

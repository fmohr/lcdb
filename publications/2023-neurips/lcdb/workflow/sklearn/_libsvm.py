from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    Float,
    Integer,
    OrConjunction,
)
from sklearn.svm import SVC

from ...utils import filter_keys_with_prefix
from .._preprocessing_workflow import PreprocessedWorkflow
import numpy as np
from scipy.special import softmax

CONFIG_SPACE = ConfigurationSpace(
    name="libsvm",
    space={
        "C": Float("C", bounds=(1e-12, 1e12), default=10**4, log=True),
        "shrinking": Categorical("shrinking", [True, False], default=True),
        "tol": Float("tol", bounds=(4.5e-5, 2), default=1e-3, log=True),
        "cap_max_iter": Categorical("cap_max_iter", [True, False], default=False),
        "max_iter": Integer("max_iter", bounds=(100, 10000), log=True, default=10000),
        "class_weight": Categorical(
            "class_weight", ["balanced", "none"], default="none"
        ),
        # "probability": Constant("probability", False),
        # "break_ties": Categorical("break_ties", [False, True], default=False), # not relevant,
    },
)

kernel = Categorical("kernel", ["poly", "rbf", "sigmoid"], default="rbf")
gamma = Float(
    "gamma",
    bounds=(1e-12, 1e12),
    default=1,
    log=True,
)
degree = Integer(
    "degree",
    bounds=(2, 5),
    default=2,
)
coef0 = Float(
    "coef0",
    bounds=(1e-12, 1e12),
    default=0.1,  # strange, 0.0 is not allowed?
    log=True,
)
CONFIG_SPACE.add_hyperparameters([kernel, degree, coef0, gamma])


class LibSVMWorkflow(PreprocessedWorkflow):
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
        C=1,
        shrinking=True,
        tol=1e-4,
        cap_max_iter=False,
        max_iter=10_000,
        class_weight="none",
        cache_size=16000.0,
        kernel="rbf",
        gamma=1,
        degree=2,
        coef0=0.1,
        random_state=None,
        **kwargs,
    ):
        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="pp@"))

        learner_kwargs = dict(
            C=C,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter if cap_max_iter else -1,
            class_weight=None if class_weight == "none" else class_weight,
            cache_size=cache_size,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            random_state=random_state,
        )
        self.learner = SVC(**learner_kwargs)

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata

        self.learner.fit(X, y)

        self.infos["classes"] = list(self.learner.classes_)
        self.infos["n_iter_"] = self.learner.n_iter_

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

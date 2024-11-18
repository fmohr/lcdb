from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
)
from sklearn.svm import SVC

from ._svm import SVMWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="libsvm",
    space={
        "C": Float("C", bounds=(1e-12, 1e12), default=1.0, log=True),
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
CONFIG_SPACE.add([kernel, degree, coef0, gamma])


class LibSVMWorkflow(SVMWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SVMWorkflow.config_space(),
    )

    def __init__(
        self,
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
        **kwargs,
    ):

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
            random_state=kwargs["random_state"] if "random_state" in kwargs else None,
        )
        svm_instance = SVC(**learner_kwargs)
        super().__init__(svm_instance, **kwargs)

    @classmethod
    def config_space(cls):
        return cls._config_space

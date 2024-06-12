from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
)

from ._base import SklearnWorkflow

from sklearn.tree import DecisionTreeClassifier

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.DTWorkflow",
    space={
        "criterion": Categorical(
            "criterion", items=["gini", "entropy", "log_loss"], default="gini"
        ),
        "min_samples_split": Integer("min_samples_split", bounds=(2, 50), default=2),
        "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 25), default=2),
        "max_features": Categorical(
            "max_features", items=["all", "sqrt", "log2"], default="sqrt"
        ),
        "min_impurity_decrease": Float(
            "min_impurity_decrease", bounds=(0.0, 1.0), default=0.0
        ),
        "bootstrap": Categorical("bootstrap", items=[True, False], default=True),
        "max_samples": Float("max_samples", bounds=(0.0, 1.0), default=1.0),
    },
)


class DTWorkflow(SklearnWorkflow):
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
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        random_state=None,
        **kwargs,
    ):

        super().__init__(
            learner=DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                random_state=random_state
            ),
            timer=timer,
            **kwargs
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

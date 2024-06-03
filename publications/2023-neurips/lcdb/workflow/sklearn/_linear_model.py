import ConfigSpace
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Integer,
    EqualsCondition,
)
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, Perceptron

from ._base import SklearnWorkflow
from ...utils import filter_keys_with_prefix, decision_fun_to_proba


CONFIG_SPACE_LR = ConfigurationSpace(
    name="sklearn.LRWorkflow",
    space={},
)


class LRWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE_LR
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        **kwargs
    ):
        super().__init__(
            learner=LogisticRegression(),
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

    @classmethod
    def config_space(cls):
        return cls._config_space


CONFIG_SPACE_RIDGE = ConfigurationSpace(
    name="sklearn.RidgeWorkflow",
    space={},
)


class RidgeWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE_RIDGE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        **kwargs
    ):
        super().__init__(
            learner=RidgeClassifier(),
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _predict_proba_after_transform(self, X):
        return decision_fun_to_proba(self.learner.decision_function(X))


CONFIG_SPACE_PA = ConfigurationSpace(
    name="sklearn.PAWorkflow",
    space={},
)


class PAWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE_PA
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        **kwargs
    ):
        super().__init__(
            learner=PassiveAggressiveClassifier(),
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _predict_proba_after_transform(self, X):
        return decision_fun_to_proba(self.learner.decision_function(X))


CONFIG_SPACE_PERCEPTRON = ConfigurationSpace(
    name="sklearn.PAWorkflow",
    space={},
)


class PerceptronWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE_PERCEPTRON
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        **kwargs
    ):
        super().__init__(
            learner=Perceptron(),
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _predict_proba_after_transform(self, X):
        return decision_fun_to_proba(self.learner.decision_function(X))

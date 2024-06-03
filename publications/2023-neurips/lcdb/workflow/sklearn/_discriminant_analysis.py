import ConfigSpace
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Integer,
    EqualsCondition,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from ._base import SklearnWorkflow
from ...utils import filter_keys_with_prefix


CONFIG_SPACE_LDA = ConfigurationSpace(
    name="sklearn.LDAWorkflow",
    space={},
)


class LDAWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE_LDA
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
            learner=LinearDiscriminantAnalysis(),
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

    @classmethod
    def config_space(cls):
        return cls._config_space


CONFIG_SPACE_QDA = ConfigurationSpace(
    name="sklearn.QDAWorkflow",
    space={},
)


class QDAWorkflow(SklearnWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE_QDA
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
            learner=QuadraticDiscriminantAnalysis(),
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

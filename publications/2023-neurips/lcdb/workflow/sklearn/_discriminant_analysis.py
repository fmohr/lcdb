from ConfigSpace import (
    ConfigurationSpace,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from ._base import SklearnWorkflow


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
        **kwargs
    ):

        super().__init__(
            learner=LinearDiscriminantAnalysis(),
            **kwargs
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return False

    @classmethod
    def is_randomizable(cls):
        return False


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
        **kwargs
    ):

        super().__init__(
            learner=QuadraticDiscriminantAnalysis(),
            **kwargs
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return False

    @classmethod
    def is_randomizable(cls):
        return False

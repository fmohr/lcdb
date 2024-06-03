import ConfigSpace
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Integer,
    EqualsCondition,
)
from sklearn.naive_bayes import GaussianNB

from ._base import SklearnWorkflow

CONFIG_SPACE_GAUSSIAN_NB = ConfigurationSpace(
    name="sklearn.GaussianNBWorkflow",
    space={},
)


class GaussianNBWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE_GAUSSIAN_NB
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
            learner=GaussianNB(),
            timer=timer,
            **kwargs
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

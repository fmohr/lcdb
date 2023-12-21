from ConfigSpace import (
    ConfigurationSpace,
    Constant,
)
from sklearn.dummy import DummyRegressor

from .._base_workflow import BaseWorkflow


CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.MedianWorkflow"
)


class MedianWorkflow(BaseWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(self, timer=None, random_state=None):
        super().__init__(timer)

        self.learner = DummyRegressor(strategy="median")

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, metadata):
        self.metadata = metadata

        self.learner.fit(X, y)

    def _predict(self, X):
        return self.learner.predict(X)

    def _predict_proba(self, X):
        raise NotImplementedError

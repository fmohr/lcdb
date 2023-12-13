from ConfigSpace import (
    ConfigurationSpace,
    Constant,
)
from sklearn.dummy import DummyClassifier

from .._base_workflow import BaseWorkflow


CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.RandomWorkflow",
    space={
        "strategy": Constant("strategy", "stratified"),
    },
)


class RandomWorkflow(BaseWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(self, timer=None, strategy="stratified", random_state=None, **kwargs):
        super().__init__(timer)

        self.learner = DummyClassifier(strategy=strategy, random_state=random_state)

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, metadata):
        self.metadata = metadata

        self.learner.fit(X, y)

        self.infos["classes"] = list(self.learner.classes_)

    def _predict(self, X):
        return self.learner.predict(X)

    def _predict_proba(self, X):
        return self.learner.predict_proba(X)

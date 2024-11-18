from ConfigSpace import (
    ConfigurationSpace,
)
from sklearn.dummy import DummyClassifier

from .._base_workflow import BaseWorkflow


CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.MajorityWorkflow"
)


class MajorityWorkflow(BaseWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(self, timer=None, **kwargs):
        super().__init__(timer, **kwargs)

        self.learner = DummyClassifier(
            strategy="prior",
            random_state=kwargs["random_state"] if "random_state" in kwargs else None
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return False

    @classmethod
    def is_randomizable(cls):
        return True

    def _fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata

        self.learner.fit(X, y)

    def _predict(self, X):
        return self.learner.predict(X)

    def _predict_proba(self, X):
        return self.learner.predict_proba(X)

from ConfigSpace import (
    ConfigurationSpace,
)
from sklearn.dummy import DummyClassifier

from .._preprocessing_workflow import PreprocessedWorkflow
from lcdb.builder.utils import filter_keys_with_prefix

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.MajorityWorkflow"
)


class MajorityWorkflowWithPreprocessing(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(self, timer=None, random_state=None, **kwargs):
        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="pp@"))

        self.learner = DummyClassifier(strategy="prior", random_state=random_state)

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata

        X = self.transform(X, y, metadata)

        self.learner.fit(X, y)

        self.infos["classes"] = list(self.learner.classes_)

    def _predict(self, X):
        return self.learner.predict(X)

    def _predict_proba(self, X):
        return self.learner.predict_proba(X)

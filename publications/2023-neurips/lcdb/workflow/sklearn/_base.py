from ConfigSpace import (
    ConfigurationSpace,
)

from lcdb.builder.utils import filter_keys_with_prefix
from .._preprocessing_workflow import PreprocessedWorkflow


CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.SklearnWorkflow",
    space={}
)


class SklearnWorkflow(PreprocessedWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    def __init__(
        self,
        learner,
        timer=None,
        **kwargs,
    ):
        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="p:pp@"))
        self.learner = learner

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata
        self.learner.fit(X, y)

    def _predict_after_transform(self, X):
        return self.learner.predict(X)

    def _predict_proba_after_transform(self, X):
        return self.learner.predict_proba(X)

    def _predict_with_proba_without_transform(self, X):
        y_pred_proba = self._predict_proba_after_transform(X)
        y_pred = y_pred_proba.argmax(axis=1)  # is based on the internal convention that labels are 0 to k-1
        return y_pred, y_pred_proba

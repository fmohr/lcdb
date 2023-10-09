import ConfigSpace
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    Integer,
)
from sklearn.neighbors import KNeighborsClassifier

from .._base_workflow import BaseWorkflow
from .._preprocessing_workflow import PreprocessedWorkflow
from ._knn_configspace import get_configspace
from ...utils import filter_keys_with_prefix


CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.LibLinearWorkflow",
    space={
        "n_neighbors": Integer("n_neighbors", (1, 100), default=5, log=True),
        "weights": Categorical("weights", ["uniform", "distance"], default="uniform"),
        "p": Integer("p", (1, 2), default=2)
    }
)


class KNNWorkflow(PreprocessedWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    def __init__(self, n_neighbors, weights, p, **kwargs):
        super().__init__(**filter_keys_with_prefix(kwargs, prefix="pp@"))

        learner_kwargs = dict(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p
        )
        self.learner = KNeighborsClassifier(**learner_kwargs)

    def update_summary(self):
        pass

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, metadata):
        self.learner.fit(X, y)
        self.infos["classes_"] = self.learner.classes_
        return self

    def _predict_proba(self, X):
        return self.learner.predict_proba(X)

    def _predict(self, X):
        return self.learner.predict(X)

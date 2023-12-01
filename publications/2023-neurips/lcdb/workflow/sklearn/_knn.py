import ConfigSpace
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Integer,
    EqualsCondition,
)
from sklearn.neighbors import KNeighborsClassifier

from .._preprocessing_workflow import PreprocessedWorkflow
from ...utils import filter_keys_with_prefix


CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.KNNWorkflow",
    space={
        "n_neighbors": Integer("n_neighbors", (1, 100), default=5, log=True),
        "weights": Categorical("weights", ["uniform", "distance"], default="uniform"),
        "p": Integer("p", (1, 10), default=2),
        # Haversine distance is only valid in 2d so we don't include it
        # Other distance could be included with more work such as Mahalanobis
        # see: https://stackoverflow.com/questions/34643548/how-to-use-mahalanobis-distance-in-sklearn-distancemetrics/34650347#34650347
        "metric": Categorical(
            "metric",
            ["minkowski", "cosine", "nan_euclidean"],
            default="minkowski",
        ),
    },
)

CONFIG_SPACE.add_conditions(
    [
        EqualsCondition(
            CONFIG_SPACE["p"],
            CONFIG_SPACE["metric"],
            "minkowski",
        ),
    ]
)


class KNNWorkflow(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        n_neighbors=5,
        weights="uniform",
        p=2,
        metric="minkowski",
        **kwargs
    ):
        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="pp@"))

        self.learner_kwargs = dict(
            n_neighbors=n_neighbors, weights=weights, p=p, metric=metric
        )
        self.learner = None

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, metadata):
        self.metadata = metadata
        X = self.transform(X, y, metadata)

        # Adapt the number of neighbors to the number of samples to avoid errors
        updated_learner_kwargs = self.learner_kwargs.copy()
        updated_learner_kwargs["n_neighbors"] = min(X.shape[0], updated_learner_kwargs["n_neighbors"])
        self.learner = KNeighborsClassifier(**updated_learner_kwargs)

        self.learner.fit(X, y)

        self.infos["classes"] = list(self.learner.classes_)

    def _predict(self, X):
        X = self.pp_pipeline.transform(X)
        return self.learner.predict(X)

    def _predict_proba(self, X):
        X = self.pp_pipeline.transform(X)
        return self.learner.predict_proba(X)

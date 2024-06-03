import ConfigSpace
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Integer,
    EqualsCondition,
)
from sklearn.neighbors import KNeighborsClassifier

from ._base import SklearnWorkflow
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
        # see: https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.DistanceMetric.html
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


class KNNWorkflow(SklearnWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
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
        super().__init__(
            learner=None,
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

        self.learner_kwargs = dict(
            n_neighbors=n_neighbors, weights=weights, p=p, metric=metric
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):

        # instantiate the learner
        updated_learner_kwargs = self.learner_kwargs.copy()
        updated_learner_kwargs["n_neighbors"] = min(X.shape[0], updated_learner_kwargs["n_neighbors"])
        self.learner = KNeighborsClassifier(**updated_learner_kwargs)

        # apply standard fitting procedure
        super()._fit_model_after_transformation(X, y, X_valid, y_valid, X_test, y_test, metadata)

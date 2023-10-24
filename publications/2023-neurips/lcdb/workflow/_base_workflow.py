import abc

from typing import Any, Tuple, Dict
from lcdb.timer import Timer
import ConfigSpace

import numpy as np

NP_ARRAY = np.ndarray


class BaseWorkflow(abc.ABC):
    def __init__(self, timer=None, **kwargs) -> None:
        super().__init__()
        self.timer = Timer() if timer is None else timer
        self.infos = {}

        # Attributes which indicates if the .transform(...) method was called
        self.transform_fitted = False
        self.workflow_fitted = False

        # Indicates if the workflow requires validation data to be fitted
        self.requires_valid_to_fit = False

        # Indicates if the workflow requires test data to be fitted (to be able to predict on test data on child fidelities)
        self.requires_test_to_fit = False

    def fit(self, X, y, metadata, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        self.timer.start("fit")
        self._fit(X=X, y=y, metadata=metadata, *args, **kwargs)
        self.timer.stop("fit")
        self.workflow_fitted = True
        return self

    @abc.abstractmethod
    def _fit(self, X, y, metadata, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> NP_ARRAY:
        """Predict from the data."""
        self.timer.start("predict")
        y_pred = self._predict(*args, **kwargs)
        self.timer.stop("predict")
        return y_pred

    def predict_proba(self, *args, **kwargs) -> NP_ARRAY:
        """Predict from the data."""
        self.timer.start("predict_proba")
        y_pred = self._predict_proba(*args, **kwargs)
        self.timer.stop("predict_proba")
        return y_pred

    @abc.abstractmethod
    def _predict(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_proba(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""
        raise NotImplementedError

    def transform(self, X, y, metadata) -> NP_ARRAY:
        """Transform the data."""
        self.timer.start("transform")
        X = self._transform(X, y, metadata)
        self.timer.stop("transform")
        self.transform_fitted = True
        return X

    @abc.abstractmethod
    def _transform(self, X, y, metadata) -> NP_ARRAY:
        """Transform the data."""
        raise NotImplementedError

    def config_space(cls) -> ConfigSpace.ConfigurationSpace:
        return cls._config_space

import abc
import time

from typing import Any, Tuple, Dict
import ConfigSpace

import numpy as np

NP_ARRAY = np.ndarray


class BaseWorkflow(abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.infos = {
            "fit_time": None,
            "predict_time": None,
        }

        # Attributes which indicates if the .transform(...) method was called
        self.transform_fitted = False
        self.workflow_fitted = False

        # Indicates if the workflow requires validation data to be fitted
        self.requires_valid_to_fit = False

        # Indicates if the workflow requires test data to be fitted (to be able to predict on test data on child fidelities)
        self.requires_test_to_fit = False

    def fit(self, X, y, metadata, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        timestamp_start = time.time()
        self._fit(X=X, y=y, metadata=metadata, *args, **kwargs)
        timestamp_end = time.time()
        self.infos["fit_time"] = timestamp_end - timestamp_start
        self.workflow_fitted = True
        return self

    @abc.abstractmethod
    def _fit(self, X, y, metadata, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> NP_ARRAY:
        """Predict from the data."""
        timestamp_start = time.time()
        y_pred = self._predict(*args, **kwargs)
        timestamp_end = time.time()
        self.infos["predict_time"] = timestamp_end - timestamp_start
        return y_pred

    @abc.abstractmethod
    def _predict(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""
        raise NotImplementedError

    def transform(self, X, y, metadata) -> NP_ARRAY:
        """Transform the data."""
        timestamp_start = time.time()
        X = self._transform(X, y, metadata)
        timestamp_end = time.time()
        self.infos["transform_time"] = timestamp_end - timestamp_start
        self.transform_fitted = True
        return X

    @abc.abstractmethod
    def _transform(self, X, y, metadata) -> NP_ARRAY:
        """Transform the data."""
        raise NotImplementedError

    def config_space(cls) -> ConfigSpace.ConfigurationSpace:
        return cls._config_space

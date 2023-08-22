import abc
import time

from typing import Any

import ConfigSpace
import numpy as np

from ..data.split import get_mandatory_preprocessing


class BaseWorkflow(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.infos = {
            "fit_time": None,
            "predict_time": None,
        }

    def fit(self, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        timestamp_start = time.time()
        self._fit(*args, **kwargs)
        timestamp_end = time.time()
        self.infos["fit_time"] = timestamp_end - timestamp_start
        return self

    @abc.abstractmethod
    def _fit(self, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> np.ndarray:
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

    def get_preprocessing_pipeline(self, X_train, y_train, binarize_sparse, drop_first):
        preprocessing_steps = get_mandatory_preprocessing(
            X_train,
            y_train,
            binarize_sparse=binarize_sparse,
            drop_first=drop_first,
            scaler=self.hyperparams["scaler"],
        )
        return preprocessing_steps

    @abc.abstractmethod
    def update_summary(self) -> dict:
        """updates the summary in case that it is not updated online."""

    @abc.abstractclassmethod
    def config_space() -> ConfigSpace.ConfigurationSpace:
        raise NotImplementedError

import abc

from typing import Any

import ConfigSpace

from ..data._split import get_mandatory_preprocessing

class BaseWorkflow(abc.ABC):
    def __init__(self, X_train, y_train, hyperparams) -> None:
        super().__init__()
        self.summary = {}
        self.hyperparams = hyperparams


    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""
        raise NotImplementedError

    def get_preprocessing_pipeline(self, X_train, y_train, binarize_sparse, drop_first):
        preprocessing_steps = get_mandatory_preprocessing(
            X_train, y_train, binarize_sparse=binarize_sparse, drop_first=drop_first, scaler=self.hyperparams['scaler']
        )
        return preprocessing_steps


    @abc.abstractmethod
    def update_summary(self) -> dict:
        """updates the summary in case that it is not updated online."""

    @abc.abstractstaticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        raise NotImplementedError


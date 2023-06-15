import abc

from typing import Any

import ConfigSpace


class BaseWorkflow(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.summary = {}

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

    @abc.abstractmethod
    def update_summary(self) -> dict:
        """updates the summary in case that it is not updated online."""

    @abc.abstractstaticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        raise NotImplementedError


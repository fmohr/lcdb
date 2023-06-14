import abc

from typing import Any


class BaseModel(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = {"scores": {}}

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> "BaseModel":
        """Fit the model to the data."""

    @abc.abstractmethod
    def predict(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""

import abc

from typing import Any
from lcdb.builder.timer import Timer
import ConfigSpace

import numpy as np
from sklearn.preprocessing import LabelEncoder

NP_ARRAY = np.ndarray


class BaseWorkflow(abc.ABC):

    def __init__(self, timer=None) -> None:
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

        self.constant_prediction = None  # this is used to treat cases where only one class is provided

        self.label_encoder = LabelEncoder()  # internally we will always work with numeric classes

    def fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata) -> "BaseWorkflow":

        # get label-encoded versions of target
        y_complete = np.concatenate([y, y_valid, y_test])
        self.label_encoder.fit(y_complete)

        self.infos["classes_overall_orig"] = self.label_encoder.classes_.tolist()

        self.timer.root["classes"] = self.infos["classes_overall_orig"]

        self.infos["classes_train_orig"] = np.unique(y).tolist()
        self.infos["classes_valid_orig"] = np.unique(y_valid).tolist()
        self.infos["classes_test_orig"] = np.unique(y_test).tolist()

        # replace labels
        y_old = y
        y = self.label_encoder.transform(y)
        y_valid = self.label_encoder.transform(y_valid)
        y_test = self.label_encoder.transform(y_test)

        # register occurring internal labels
        self.infos["classes_overall"] = list(range(len(self.label_encoder.classes_)))
        self.infos["classes_train"] = np.unique(y).tolist()
        self.infos["classes_valid"] = np.unique(y_valid).tolist()
        self.infos["classes_test"] = np.unique(y_test).tolist()

        # if there is only one class, create a dummy classifier and always return this class
        if len(self.infos["classes_train"]) == 1:
            self.constant_prediction = self.infos["classes_train"][0]
        else:
            self.constant_prediction = None

            """Fit the workflow to the data."""
            with self.timer.time("fit"):
                self._fit(
                    X=X,
                    y=y,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=X_test,
                    y_test=y_test,
                    metadata=metadata
                )
        self.workflow_fitted = True
        return self

    @abc.abstractmethod
    def _fit(self, X, y, metadata, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> NP_ARRAY:
        """Predict from the data."""
        with self.timer.time("predict"):
            y_pred = self.get_predictions_from_probas(self.predict_proba(*args, **kwargs))
            raise Exception(y_pred)

        return y_pred

    def get_predictions_from_probas(self, y_proba):
        return self.label_encoder.inverse_transform(np.argmax(y_proba, axis=1))

    @abc.abstractmethod
    def _predict(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""
        raise NotImplementedError

    def predict_proba(self, *args, **kwargs) -> NP_ARRAY:
        """Predict from the data."""
        with self.timer.time("predict_proba"):

            # if there is a constant prediction, do not refer to the actual workflow
            if self.constant_prediction is not None:
                X = kwargs["X"] if "X" in kwargs else args[0]
                y_pred = np.ones(X.shape[0])
            else:
                y_pred = self._predict_proba(*args, **kwargs)
        return y_pred

    @abc.abstractmethod
    def _predict_proba(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""
        raise NotImplementedError

    def transform(self, X, y, metadata, timer_suffix="") -> NP_ARRAY:
        """Transform the data."""
        with self.timer.time("transform" + timer_suffix):
            X = self._transform(X, y, metadata)
        self.transform_fitted = True
        return X

    def _transform(self, X, y, metadata) -> NP_ARRAY:
        """Transform the data."""
        return X

    @classmethod
    def config_space(cls) -> ConfigSpace.ConfigurationSpace:
        return cls._config_space

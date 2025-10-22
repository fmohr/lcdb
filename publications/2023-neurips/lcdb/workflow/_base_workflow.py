import abc
import logging

from typing import Any
from lcdb.builder.timer import Timer
from lcdb.builder.utils import get_random_state
import ConfigSpace

import numpy as np
from sklearn.preprocessing import LabelEncoder

NP_ARRAY = np.ndarray


class BaseWorkflow(abc.ABC):

    def __init__(self, n_jobs=1, timer=None, logger=None, random_state=None, memory_limit_in_bytes=None) -> None:
        super().__init__()
        if timer is None:
            self.timer = Timer()
        elif not isinstance(timer, Timer):
            raise ValueError(f"timer must be None or object of Timer but is {type(timer)}")
        else:
            self.timer = timer
        
        self.logger = logging.getLogger("LCDB") if logger is None else logger
        
        # set memory limit
        self.memory_limit_in_bytes = memory_limit_in_bytes
        self.n_jobs = n_jobs
        if not self.is_parallelizable() and n_jobs > 1:
            logger.warning(
                f"Workflow {self.__class__.__name__} is not parallelizable but n_jobs was set to {n_jobs}!"
            )

        # generate warning if the randomness is not seeded
        if self.__class__.is_randomizable() and random_state is None:
            logger.warning(
                f"Workflow {self.__class__.__name__} is randomizable but no seed was provided!"
            )
        self.random_state = get_random_state(random_state)

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
    
    def get_reason_why_workflow_cannot_be_built_at_anchor(self, X, y, a):
        """
            Determines whether the workflow can be fitted on given data at a specific anchor.
            This should only be called if it is known already that the workflow is generally fittable on the dataset.

            :return: `None` if the workflow can be built at the given anchor. Otherwise a string with the reason why it cannot.
        """
        return None
    
    def fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata) -> "BaseWorkflow":

        # check that categorial columns are properly encoded in metadata
        if type(metadata["categories"]["columns"]) != np.ndarray or len(metadata["categories"]["columns"].shape) != 1:
            raise ValueError(f"The binary mask for categorical columns must be a binary 1D numpy array but is {type(metadata['categories']['columns'])}")

        # get label-encoded versions of target
        y_complete = np.concatenate([y, y_valid, y_test])
        self.label_encoder.fit(y_complete)

        self.infos["classes_overall_orig"] = self.label_encoder.classes_.tolist()

        self.timer.root["classes"] = self.infos["classes_overall_orig"]

        self.infos["classes_train_orig"] = np.unique(y).tolist()
        self.infos["classes_valid_orig"] = np.unique(y_valid).tolist()
        self.infos["classes_test_orig"] = np.unique(y_test).tolist()

        # replace labels
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

    @classmethod
    @abc.abstractmethod
    def builds_iteration_curve(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def is_randomizable(cls):
        pass

    @classmethod
    def is_parallelizable(cls):
        return False

    @abc.abstractmethod
    def _fit(self, X, y, metadata, *args, **kwargs) -> "BaseWorkflow":
        """Fit the workflow to the data."""
        pass

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
        """
            Predict probabilities from the data.
            The number of columns returned by this method is always the number of classes in the dataset,
            even if the workflow has not seen all classes during training.
        """
        with self.timer.time("predict_proba"):

            # if there is a constant prediction, do not refer to the actual workflow
            if self.constant_prediction is not None:
                self.logger.info(f"Only one class is known, so predicting only label {self.infos['classes_train'][0]} on all instances, without even looking at the model.")
                X = kwargs["X"] if "X" in kwargs else args[0]
                y_pred = np.ones((X.shape[0], 1))
            else:
                y_pred = self._predict_proba(*args, **kwargs)
            
            # if the model has not seen all labels, extend the prediction matrix respectively
            if len(self.infos["classes_train"]) < len(self.infos["classes_overall"]):
                X = kwargs["X"] if "X" in kwargs else args[0]
                y_pred_extended = np.zeros((X.shape[0], len(self.infos["classes_overall"])))
                y_pred_extended[:, self.infos["classes_train"]] = y_pred
                self.logger.info(f"Extended output shape from originally {y_pred.shape} to {y_pred_extended.shape}")
                y_pred = y_pred_extended

        assert np.all(np.isclose(y_pred.sum(axis=1), 1.0)), f"inconsistent probabilities, sums are: {y_pred.sum(axis=1)}"
        return y_pred

    @abc.abstractmethod
    def _predict_proba(
        self, *args, **kwargs
    ) -> Any:  # TODO: not sure what the return type should be
        """Predict from the data."""
        raise NotImplementedError

    def transform(self, X, y, metadata, timer_suffix="") -> NP_ARRAY:
        """Transform the data."""
        with self.timer.time("transform" + timer_suffix) as timer:
            X = self._transform(X, y, metadata)
        self.transform_fitted = True
        return X

    def _transform(self, X, y, metadata) -> NP_ARRAY:
        """Transform the data."""
        return X

    @classmethod
    def config_space(cls) -> ConfigSpace.ConfigurationSpace:
        return cls._config_space

import numpy as np

from ConfigSpace import (
    ConfigurationSpace,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from ._base import SklearnWorkflow


CONFIG_SPACE_LDA = ConfigurationSpace(
    name="sklearn.LDAWorkflow",
    space={},
)


class LDAWorkflow(SklearnWorkflow):

    # Static Attribute
    _config_space = CONFIG_SPACE_LDA
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            learner=LinearDiscriminantAnalysis(),
            **kwargs
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return False

    @classmethod
    def is_randomizable(cls):
        return False


CONFIG_SPACE_QDA = ConfigurationSpace(
    name="sklearn.QDAWorkflow",
    space={},
)


class QDAWorkflow(SklearnWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE_QDA
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=SklearnWorkflow.config_space(),
    )

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            learner=QuadraticDiscriminantAnalysis(),
            **kwargs
        )

        self.threat_of_instability = None

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return False

    @classmethod
    def is_randomizable(cls):
        return False

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        
        self.threat_of_instability = False
        for label, cnts in zip(*np.unique(y, return_counts=True)):
            if cnts < X.shape[1]:
                self.logger.warning(
                    f"Insufficient data points for class {label}. Has only {cnts} samples, but the data has {X.shape[1]} features. "
                    "This is a threat for stability of QDA. Possible nans in predictions will be replaced by uniform probabilities."
                )
                self.threat_of_instability = True
        super()._fit_model_after_transformation(X, y, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test, metadata=metadata)

    def _predict_proba_after_transform(self, X):
        y_pred_proba = super()._predict_proba_after_transform(X)
        if self.threat_of_instability:
            rows_with_nans = np.any(np.isnan(y_pred_proba), axis=1)
            num_rows_with_nans = np.count_nonzero(rows_with_nans)
            if num_rows_with_nans > 0:
                self.logger.warning(f"Correcting the predictions of {num_rows_with_nans}/{X.shape[0]} instances with nans to uniform prediction.")
                y_pred_proba[rows_with_nans] = 1 / len(self.infos["classes_train"])
        return y_pred_proba
    

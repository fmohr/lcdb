import ConfigSpace

from ._knn_configspace import get_configspace
from .._base_workflow import BaseWorkflow
import os
from .._util import unserialize_config_space
from sklearn.neighbors import KNeighborsClassifier
class KNNWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__(X_train, y_train, hyperparams)
        hyperparams = hyperparams.copy()
        hyperparams.pop('scaler')
        self.learner = KNeighborsClassifier(**hyperparams)

    def update_summary(self):
        pass

    @staticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        return get_configspace()

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        self.learner.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.learner.predict_proba(X)

    def predict(self, X):
        return self.learner.predict(X)
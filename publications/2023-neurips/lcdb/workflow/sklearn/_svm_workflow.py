import ConfigSpace

from .._base_workflow import BaseWorkflow
import os
from .._util import unserialize_config_space
from sklearn.svm import SVC
class SVMWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__()
        hyperparams['verbose'] = True
        self.learner = SVC(**hyperparams)

    def update_summary(self):
        pass

    @staticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        path = os.path.abspath(__file__)
        path = path[:path.rfind(os.sep) + 1]
        return unserialize_config_space(path + "_svm_cs.json")

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        self.learner.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        raise Exception("Sorry, we don't do probabilistic classification with SVMs")

    def predict(self, X):
        return self.learner.predict(X)
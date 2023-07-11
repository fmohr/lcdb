import ConfigSpace

from .._base_workflow import BaseWorkflow
import os
from .._util import unserialize_config_space
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

class LibLinearWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__()

        if hyperparams['class_weight'] == 'none':
            hyperparams['class_weight'] = None

        if hyperparams["multiclass"] == "ovr":
            del hyperparams["multiclass"]
            hyperparams["multi_class"] = "ovr"
            self.learner = LinearSVC(**hyperparams)
            return
        if hyperparams["multiclass"] == "ovo-scikit":
            del hyperparams["multiclass"]
            hyperparams["multi_class"] = "ovr"
            self.learner = OneVsOneClassifier(LinearSVC(**hyperparams), n_jobs=None)
            return

        raise Exception("Multiclass strategy not implemented")


    def update_summary(self):
        pass

    @staticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        path = os.path.abspath(__file__)
        path = path[:path.rfind(os.sep) + 1]
        return unserialize_config_space(path + "_liblinear_cs.json")

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        self.learner.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        raise Exception("Sorry, we don't do probabilistic classification with SVMs")

    def predict(self, X):
        return self.learner.predict(X)
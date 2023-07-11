import ConfigSpace

from .._base_workflow import BaseWorkflow
import os
from .._util import unserialize_config_space
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

class LibSVMWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__()

        if hyperparams["cap_max_iter"]:
            pass # do nothing, max_iter already set
        else:
            hyperparams["max_iter"] = 1
        del hyperparams["cap_max_iter"]

        if hyperparams['class_weight'] == 'none':
            hyperparams['class_weight'] = None

        if hyperparams["multiclass"] == "ovo":
            del hyperparams["multiclass"]
            hyperparams["decision_function_shape"] = "ovo"
            self.learner = SVC(**hyperparams)
            return
        if hyperparams["multiclass"] == "ovr-scikit":
            del hyperparams["multiclass"]
            self.learner = OneVsRestClassifier(SVC(**hyperparams), n_jobs=None, verbose=50)
            return

        raise Exception("Multiclass strategy not implemented")


    def update_summary(self):
        pass

    @staticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        path = os.path.abspath(__file__)
        path = path[:path.rfind(os.sep) + 1]
        return unserialize_config_space(path + "_libsvm_cs.json")

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        self.learner.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        raise Exception("Sorry, we don't do probabilistic classification with SVMs")

    def predict(self, X):
        return self.learner.predict(X)
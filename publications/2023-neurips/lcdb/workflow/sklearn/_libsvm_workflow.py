import ConfigSpace

from .._base_workflow import BaseWorkflow
import os
from .._util import unserialize_config_space
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from ._libsvm_congfigspace import get_configspace

class LibSVMWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__(X_train, y_train, hyperparams)

        hyperparams = hyperparams.copy()
        hyperparams['verbose'] = True
        hyperparams.pop('scaler')

        if hyperparams["cap_max_iter"]:
            pass # do nothing, max_iter already set
        else:
            hyperparams["max_iter"] = -1
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
        return get_configspace()

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        self.learner.fit(X_train, y_train)
        if type(self.learner) is SVC:
            self.summary["n_iter_"] = self.learner.n_iter_.tolist()
            self.summary["n_support_"] = self.learner.n_support_.tolist()
        else:
            n_iter_ = []
            n_support_ = []
            for est in self.estimators_:
                n_iter_.append(est.n_iter_)
                n_support_.append(est.n_support_)
            self.summary["n_iter_"] = n_iter_
            self.summary["n_support_"] = n_support_
        return self

    def predict_proba(self, X):
        raise Exception("Sorry, we don't do probabilistic classification with SVMs")

    def predict(self, X):
        return self.learner.predict(X)
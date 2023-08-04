import ConfigSpace

from .._base_workflow import BaseWorkflow
import os
from .._util import unserialize_config_space
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

from ._liblinear_configspace import get_configspace

class LibLinearWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__(X_train, y_train, hyperparams)

        hyperparams = hyperparams.copy()
        hyperparams['verbose'] = True
        hyperparams.pop('scaler')

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
        return get_configspace()

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        self.learner.fit(X_train, y_train)

        if type(self.learner) is LinearSVC:
            self.summary["n_iter_"] = self.learner.n_iter_
        else:
            n_iter_ = []
            for est in self.learner.estimators_:
                n_iter_.append(est.n_iter_)
            self.summary["n_iter_"] = n_iter_

        return self

    def predict_proba(self, X):
        raise Exception("Sorry, we don't do probabilistic classification with SVMs")

    def predict(self, X):
        return self.learner.predict(X)

    def decision_function(self, X):
        return self.learner.decision_function(X)
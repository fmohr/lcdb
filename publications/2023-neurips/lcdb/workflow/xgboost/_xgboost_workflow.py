import ConfigSpace

from lcdb.workflow._base_workflow import BaseWorkflow
# from .._base_workflow import BaseWorkflow
import os
# from .._util import unserialize_config_space
from lcdb.workflow._util import unserialize_config_space
from xgboost import XGBClassifier

class XGBoostWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__()
        self.learner = XGBClassifier(**hyperparams)

    def update_summary(self):
        pass

    @staticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        # TODO: Better to create the configspace in Python directly, see _knn_workflow.py
        path = os.path.abspath(__file__)
        path = path[:path.rfind(os.sep) + 1]
        return unserialize_config_space(path + "_xgboost_cs.json")

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        self.learner.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.learner.predict_proba(X)

    def predict(self, X):
        return self.learner.predict(X)


if __name__ == "__main__":
    from lcdb.data._sklearn import _load_iris
    (X, y), metadata = _load_iris()
    cs = XGBoostWorkflow.get_config_space()
    config = cs.sample_configuration(1).get_dictionary()
    xgb = XGBoostWorkflow(X_train=X, y_train=y, hyperparams=config)
    xgb.fit((X, y), None, None)
    xgb.predict(X)
    xgb.predict_proba(X)

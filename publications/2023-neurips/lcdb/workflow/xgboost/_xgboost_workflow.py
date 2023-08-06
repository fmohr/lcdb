import ConfigSpace

from .._base_workflow import BaseWorkflow
#from lcdb.workflow._base_workflow import BaseWorkflow
from xgboost import XGBClassifier
from ._xgboost_configspace import get_configspace
#from _xgboost_configspace import get_configspace

class XGBoostWorkflow(BaseWorkflow):

    def __init__(self, X_train, y_train, hyperparams):
        super().__init__(X_train, y_train, hyperparams)
        hyperparams = hyperparams.copy()
        hyperparams.pop("scaler")
        self.learner = XGBClassifier(**hyperparams)

    def update_summary(self):
        pass

    @staticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        return get_configspace()

    def fit(self, data_train, data_valid, data_test) -> "BaseWorkflow":
        X_train, y_train = data_train
        # FIXME: xgboost requires integer values for the class labels [0, n_classes-1]
        y_train = y_train.astype(int)  # FIXME: this should be fixed in the preprocessing
        self.learner.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        # FIXME: class labels?
        return self.learner.predict_proba(X)

    def predict(self, X):
        # FIXME: class labels?
        y = self.learner.predict(X)
        return y.astype(str)


if __name__ == "__main__":
    import numpy as np
    #from lcdb.data._sklearn import _load_iris
    from lcdb.data._openml import get_openml_dataset
    #(X, y), metadata = _load_iris()
    openmlid = 6
    binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
    drop_first = False  # openmlid not in [3] # drop first cannot be used in datasets with some very rare categorical values
    X, y = get_openml_dataset(openmlid)
    y = np.array([str(e) for e in y])  # make sure that labels are strings
    cs = XGBoostWorkflow.get_config_space()
    config = cs.sample_configuration(1).get_dictionary()
    xgb = XGBoostWorkflow(X_train=X, y_train=y, hyperparams=config)
    xgb.fit((X, y), None, None)
    xgb.predict(X)
    xgb.predict_proba(X)

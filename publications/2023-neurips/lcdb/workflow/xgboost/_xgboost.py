from xgboost import XGBClassifier, DMatrix
from xgboost.callback import TrainingCallback
from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform
#from lcdb.utils import filter_keys_with_prefix
#from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from ...utils import filter_keys_with_prefix
from .._preprocessing_workflow import PreprocessedWorkflow
from sklearn.preprocessing import LabelEncoder
import time
import numpy as np

CONFIG_SPACE = ConfigurationSpace(
    name="xgboost",
    space={
        # FIMXE: once we integrate the EvalCallBack and/or do early stopping we can simply set n_estimators to a high value
        #"n_estimators": Integer("n_estimators", bounds=(1, 2**12), distribution=Uniform(), default=100, log=True),
        "learning_rate": Float(
            "learning_rate",
            bounds=(10**-6, 1),
            distribution=Uniform(),
            default=0.3,
            log=True,
        ),
        "gamma": Float(
            "gamma",
            bounds=(10**-6, 2**6),
            distribution=Uniform(),
            default=10**-6,
            log=True,
        ),  # normal default would be 0
        "min_child_weight": Float(
            "min_child_weight",
            bounds=(10**-6, 2**5),
            distribution=Uniform(),
            default=1,
            log=True,
        ),
        "max_depth": Integer(
            "max_depth", bounds=(2, 2**5), distribution=Uniform(), default=6, log=True
        ),
        "subsample": Float(
            "subsample", bounds=(0.5, 1), distribution=Uniform(), default=1
        ),
        "colsample_bytree": Float(
            "colsample_bytree", bounds=(0.3, 1), distribution=Uniform(), default=1
        ),
        "reg_alpha": Float(
            "reg_alpha",
            bounds=(10**-6, 2),
            distribution=Uniform(),
            default=10**-6,
            log=True,
        ),  # normal default would be 0
        "reg_lambda": Float(
            "reg_lambda",
            bounds=(10**-6, 2),
            distribution=Uniform(),
            default=1,
            log=True,
        ),
    },
)


class ExtendedLabelEncoder(LabelEncoder):
    def fit(self, y):
        super(ExtendedLabelEncoder, self).fit(y)
        self.classes_ = np.append(self.classes_, "UNKNOWN")
        return self

    def transform(self, y):
        new_y = np.array([label if label in self.classes_ else "UNKNOWN" for label in y])
        return super(ExtendedLabelEncoder, self).transform(new_y)


# FIXME: reiterate
class TimingCallback(TrainingCallback):
    def __init__(self):
        self.start_time = None
        self.times = []

    def before_iteration(self, model, epoch, evals_log):
        self.start_time = time.time()

    def after_iteration(self, model, epoch, evals_log):
        end_time = time.time()
        iteration_time = end_time - self.start_time
        # FIXME: this measures the time for the whole iteration, not just the training
        # also this is the time for the single interation not the overall time
        self.times.append(iteration_time)


# FIXME: reiterate
class EvalCallBack(TrainingCallback):
    #def __init__(self, data_train, data_valid, data_test, learner):
    def __init__(self, data_train, data_valid, learner):
        self.DM_train = DMatrix(data_train[0], label=data_train[1])
        self.y_train = data_train[1]
        self.DM_valid = DMatrix(data_valid[0], label=data_valid[1])
        self.y_valid = data_valid[1]
        #self.DM_test = DMatrix(data_test[0], label=data_test[1])
        #self.y_test = data_test[1]
        self.n_classes = len(np.unique(self.y_train))
        self.metrics_train = []
        self.metrics_valid = []
        #self.metrics_test = []
        self.metric_ids = ["accuracy"]

    def after_iteration(self, model, epoch, evals_log):
        # predict on train, valid, test
        probs_pred_train = model.predict(self.DM_train, strict_shape=True)
        probs_pred_train = self.create_full_probs(probs_pred_train, n_classes=self.n_classes)
        y_pred_train = self.create_labels_from_probs(probs_pred_train)

        probs_pred_valid = model.predict(self.DM_valid, strict_shape=True)
        probs_pred_valid = self.create_full_probs(probs_pred_valid, n_classes=self.n_classes)
        y_pred_valid = self.create_labels_from_probs(probs_pred_valid)

        #probs_pred_test = model.predict(self.DM_test, strict_shape=True)
        #probs_pred_test = self.create_full_probs(probs_pred_test, n_classes=self.n_classes)
        #y_pred_test = self.create_labels_from_probs(probs_pred_test)

        # calculate the metrics
        # FIXME: accuracy, ...
        acc_train = np.mean(y_pred_train == self.y_train)
        acc_valid = np.mean(y_pred_valid == self.y_valid)
        #acc_test = np.mean(y_pred_test == self.y_test)

        # store the metrics
        self.metrics_train.append({epoch: [acc_train]})
        self.metrics_valid.append({epoch: [acc_valid]})
        #self.metrics_test.append({epoch: [acc_test]})
        return False

    def create_full_probs(self, probs_pred, n_classes):
        if n_classes == 2:
            # add the first class (0) to the probabilities
            probs_pred = np.concatenate((1 - probs_pred, probs_pred), axis=1)
        return probs_pred

    def create_labels_from_probs(self, probs_pred):
        # get the index of the highest probability
        y_pred = np.argmax(probs_pred, axis=1)
        return y_pred


class XGBoostWorkflow(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    def __init__(self, n_estimators=100, learning_rate=0.3, gamma=10**-6, min_child_weight=0, max_depth=6, subsample=1, colsample_bytree=1, reg_alpha=10**-6, reg_lambda=1, random_state=None, **kwargs):

        super().__init__(**filter_keys_with_prefix(kwargs, prefix="pp@"))

        self.requires_valid_to_fit = True
        self._transformer_label = ExtendedLabelEncoder()
        # FIXME: early stopping is not supported yet but we could do this
        # either based on the passed validation data or letting XGBoost take its own slice from the training data
        # probably we want to try out both

        self.fidelities = {
            "fidelity_unit": "iterations",
            "fidelity_values": [],
            "score_types": ["accuracy"],
            "score_values": [],
            "time_types": ["iteration"],
            "time_values": [],
        }

        learner_kwargs = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )
        self.learner = XGBClassifier(**learner_kwargs)

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, X_valid, y_valid, metadata):
        # FIXME: what we actually want to do is use the custom EvalCallBack from above to log metrics over the boosting iterations for all splits
        # however, we then require X_valid, y_valid, AND X_test, y_test to be passed to the fit method which is currently not supported
        # Also: this has some effect on the time we measure and we need to think about how to handle this
        self.metadata = metadata
        X = self.transform(X, y, metadata)
        y = self._transformer_label.fit_transform(y)
        X_valid = self.transform(X_valid, y_valid, metadata)
        y_valid = self._transformer_label.transform(y_valid)
        eval_callback = EvalCallBack((X, y), data_valid=(X_valid, y_valid), learner=self.learner)
        timing_callback = TimingCallback()

        self.learner.fit(X, y, callbacks=[eval_callback, timing_callback])

        # Collect metrics
        # FIXME:
        # fidelity_unit: a string, describing the unit of the fidelity (e.g., samples, epochs, batches, resolution, etc.).
        # fidelity_values: a 1-D array of reals, giving the fidelity value at which the score_values and time_values are collected.
        # score_types: a 1-D array of strings, describing the name(s) of the scoring function(s) collected (e.g., loss, accuracy, balanced_accuracy, etc.).
        # score_values: a 3-D array of reals, where axis=0 corresponds to fidelity_values and has the same length, where axis=1 corresponds to data splits ["train", "valid", "test"] (the 3 are not always present) with a length from 1 to 3, where axis=2 corresponds to score_types and has the same length.
        # time_types: a 1-D array of strings, where each value describes a type of timing (e.g., fit, predict, epoch).
        # times_values: a 3-D array of reals, where axis=0 corresponds to fidelity_values and has the same length, where axis=1 corresponds to data splits ["train", "valid", "test"] (the 3 are not always present) with a length from 1 to 3, where axis=2 corresponds to time_types and has the same length. REFINE

        # FIXME: check if score_types and callback metric_ids are the same
        self.fidelities["fidelity_values"] = [i for i in range(1, len(timing_callback.times) + 1)]
        self.fidelities["score_values"] = [[list(eval_callback.metrics_train[i][i]), list(eval_callback.metrics_valid[i][i])] for i in range(self.fidelities["fidelity_values"][-1])]
        self.fidelities["time_values"] = timing_callback.times

    def _predict(self, X):
        X_pred = self.pp_pipeline.transform(X)
        y_pred = self.learner.predict(X_pred)
        return self._transformer_label.inverse_transform(y_pred)

#if __name__ == "__main__":
#    # Just for quick testing
#    import numpy as np
#    from lcdb.data import load_task
#    from lcdb.data.split import train_valid_test_split
#    from sklearn.preprocessing import OneHotEncoder
#    from sklearn.preprocessing import FunctionTransformer
#    openml_id = 6
#    known_categories = True
#    test_seed = 0
#    valid_seed = 0
#    test_prop = 0.2
#    valid_prop = 0.2
#    (X, y), dataset_metadata = load_task(f"openml.{openml_id}")
#    num_instances = X.shape[0]
#    columns_categories = np.asarray(dataset_metadata["categories"], dtype=bool)
#    values_categories = None
#    dataset_metadata["categories"] = {"columns": columns_categories}
#    if not (np.any(columns_categories)):
#        one_hot_encoder = FunctionTransformer(func=lambda x: x, validate=False)
#    else:
#        dataset_metadata["categories"]["values"] = None
#        one_hot_encoder = OneHotEncoder(
#            drop="first", sparse_output=False
#        )  # TODO: drop "first" could be an hyperparameter
#        one_hot_encoder.fit(X[:, columns_categories])
#        if known_categories:
#            values_categories = one_hot_encoder.categories_
#            values_categories = [v.tolist() for v in values_categories]
#            dataset_metadata["categories"]["values"] = values_categories
#    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y, test_seed, valid_seed, test_prop, valid_prop, stratify=True)
#    workflow = XGBoostWorkflow()
#    workflow.fit(X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid, metadata=dataset_metadata)
#    y_pred = workflow.predict(X_test)
#    print(np.mean(y_pred == y_test))

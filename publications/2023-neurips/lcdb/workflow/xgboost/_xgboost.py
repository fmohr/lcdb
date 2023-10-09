from xgboost import XGBClassifier, DMatrix
from xgboost.callback import TrainingCallback
from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform, Categorical
#from lcdb.utils import filter_keys_with_prefix
#from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from ...utils import filter_keys_with_prefix
from .._preprocessing_workflow import PreprocessedWorkflow
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.preprocessing import LabelEncoder
import time
import numpy as np


CONFIG_SPACE = ConfigurationSpace(
    name="xgboost",
    space={
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
        "early_stopping": Categorical(
            "early_stopping",
            items=[True, False],
            default=False,
        )
    },
)


# FIXME: reiterate to perform early stopping based on a metric and early stopping rounds in this callback
class EvalCallBack(TrainingCallback):
    def __init__(self, data_train, data_valid, data_test, encoder, early_stopping, early_stopping_rounds, early_stopping_metric):
        self.encoder = encoder
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_counter = 0
        self.DM_train = DMatrix(data_train[0], label=data_train[1])
        self.y_train = data_train[1]
        self.DM_valid = DMatrix(data_valid[0])
        self.y_valid = data_valid[1]
        self.DM_test = DMatrix(data_test[0])
        self.y_test = data_test[1]
        self.n_classes = len(np.unique(self.y_train))
        self.metrics_train = []
        self.metrics_valid = []
        self.metrics_test = []
        self.metric_ids = ["accuracy", "loss"]
        self.iterations = []
        self.time_before_iterations = []
        self.time_after_iterations = []
        self.time_train = []
        self.time_predict_train = []
        self.time_predict_valid = []
        self.time_predict_test = []
        self.time_overhead = []

    def before_iteration(self, model, epoch, evals_log):
        # calculate the time
        self.time_before_iterations.append(time.time())
        return False

    def after_iteration(self, model, epoch, evals_log):
        # update the iteration
        self.iterations.append(epoch + 1)  # epoch starts at 0

        # calculate the time
        self.time_train.append(time.time() - self.time_before_iterations[-1])

        # FIXME: measure predict time after labels are created vs before labels are created
        #        also include time for metric calculation and storage?
        # predict on train, valid, test and calculate the metrics and store them
        # FIXME: accuracy, 0-1 loss, ...
        # FIXME: we should track the time needed to calculate each metric separately to simulate HPO
        time_before_predict_train = time.time()
        probs_pred_train = model.predict(self.DM_train, strict_shape=True)
        probs_pred_train = self.create_full_probs(probs_pred_train, n_classes=self.n_classes)
        y_pred_train = self.create_labels_from_probs(probs_pred_train)
        acc_train = accuracy_score(self.encoder.inverse_transform(self.y_train), self.encoder.inverse_transform(y_pred_train))
        loss_train = zero_one_loss(self.encoder.inverse_transform(self.y_train), self.encoder.inverse_transform(y_pred_train))
        self.metrics_train.append([acc_train, loss_train])
        self.time_predict_train.append(time.time() - time_before_predict_train)

        time_before_predict_valid = time.time()
        probs_pred_valid = model.predict(self.DM_valid, strict_shape=True)
        probs_pred_valid = self.create_full_probs(probs_pred_valid, n_classes=self.n_classes)
        y_pred_valid = self.create_labels_from_probs(probs_pred_valid)
        acc_valid = accuracy_score(self.y_valid, self.encoder.inverse_transform(y_pred_valid))
        loss_valid = zero_one_loss(self.y_valid, self.encoder.inverse_transform(y_pred_valid))
        self.metrics_valid.append([acc_valid, loss_valid])
        self.time_predict_valid.append(time.time() - time_before_predict_valid)

        time_before_predict_test = time.time()
        probs_pred_test = model.predict(self.DM_test, strict_shape=True)
        probs_pred_test = self.create_full_probs(probs_pred_test, n_classes=self.n_classes)
        y_pred_test = self.create_labels_from_probs(probs_pred_test)
        acc_test = accuracy_score(self.y_test, self.encoder.inverse_transform(y_pred_test))
        loss_test = zero_one_loss(self.y_test, self.encoder.inverse_transform(y_pred_test))
        self.metrics_test.append([acc_test, loss_test])
        self.time_predict_test.append(time.time() - time_before_predict_test)

        self.time_after_iterations.append(time.time())

        # check if early stopping is necessary
        if self.early_stopping:
            if self.iterations[-1] == 1:
                return False
            if self.early_stopping_metric == "accuracy":
                # increment the counter if the validation accuracy has not improved otherwise set it to 0
                if self.metrics_valid[-1][0] <= self.metrics_valid[-2][0]:
                    self.early_stopping_counter += 1
                else:
                    self.early_stopping_counter = 0
                # stop if the counter is greater than the early stopping rounds
                if self.early_stopping_counter > self.early_stopping_rounds:
                    return True
            else:
                raise NotImplementedError
        return False

    def create_full_probs(self, probs_pred, n_classes):
        if n_classes == 2:
            # add the first class (0) to the probabilities
            probs_pred = np.concatenate((1 - probs_pred, probs_pred), axis=1)
        return probs_pred

    def create_labels_from_probs(self, probs_pred, invert: bool = False):
        # get the index of the highest probability
        y_pred = np.argmax(probs_pred, axis=1)
        # invert the label encoding index if necessary
        if invert:
            y_pred = self.encoder.inverse_transform(y_pred)
        return y_pred


class XGBoostWorkflow(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    # FIXME: increase the number of iterations to something like 1000-5000
    # FIXME: random_state
    # FIXME: trycatch
    def __init__(self, n_estimators=100, learning_rate=0.3, gamma=10**-6, min_child_weight=0, max_depth=6, subsample=1, colsample_bytree=1, reg_alpha=10**-6, reg_lambda=1, early_stopping=True, early_stopping_rounds=10, early_stopping_metric="accuracy", random_state=None, **kwargs):

        super().__init__(**filter_keys_with_prefix(kwargs, prefix="pp@"))

        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric

        self.fidelities = {
            "fidelity_unit": "iterations",
            "fidelity_values": [],
            "score_types": ["accuracy", "loss"],
            "score_values": [],
            "time_types": ["train_seconds_iteration", "predict_seconds_iteration"],
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

        self.encoder = LabelEncoder()

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        start_time = time.time()
        self.metadata = metadata
        y = self.encoder.fit_transform(y)
        time_before_transform = time.time()
        X = self.transform(X, y, metadata)
        time_transform_train = time.time() - time_before_transform
        X_valid = self.transform(X_valid, y_valid, metadata)
        time_transform_valid = time.time() - time_before_transform - time_transform_train
        X_test = self.transform(X_test, y_test, metadata)
        time_transform_test = time.time() - time_before_transform - time_transform_train - time_transform_valid

        eval_callback = EvalCallBack((X, y), data_valid=(X_valid, y_valid), data_test=(X_test, y_test), encoder=self.encoder, early_stopping=self.early_stopping, early_stopping_rounds=self.early_stopping_rounds, early_stopping_metric="accuracy")
        self.learner.set_params(callbacks=[eval_callback])

        print("Prepare and Transform overhead:")
        print(time.time() - start_time)

        time_before_fit = time.time()
        self.learner.fit(X, y)
        time_after_fit = time.time()

        print("Actual fit time:")
        print(time_after_fit - time_before_fit)

        # Collect metrics
        # fidelity_unit: a string, describing the unit of the fidelity (e.g., samples, epochs, batches, resolution, etc.).
        # fidelity_values: a 1-D array of reals, giving the fidelity value at which the score_values and time_values are collected.
        # score_types: a 1-D array of strings, describing the name(s) of the scoring function(s) collected (e.g., loss, accuracy, balanced_accuracy, etc.).
        # score_values: a 3-D array of reals, where axis=0 corresponds to fidelity_values and has the same length, where axis=1 corresponds to data splits ["train", "valid", "test"] (the 3 are not always present) with a length from 1 to 3, where axis=2 corresponds to score_types and has the same length.
        # time_types: a 1-D array of strings, where each value describes a type of timing (e.g., fit, predict, epoch).
        # times_values: a 3-D array of reals, where axis=0 corresponds to fidelity_values and has the same length, where axis=1 corresponds to data splits ["train", "valid", "test"] (the 3 are not always present) with a length from 1 to 3, where axis=2 corresponds to time_types and has the same length. REFINE
        self.fidelities["fidelity_values"] = eval_callback.iterations
        self.fidelities["score_values"] = [[list(eval_callback.metrics_train[i]), list(eval_callback.metrics_valid[i]), list(eval_callback.metrics_test[i])] for i in range(self.fidelities["fidelity_values"][-1])]
        self.fidelities["time_values"] = [[[eval_callback.time_train[i], eval_callback.time_predict_train[i]], ["NaN", eval_callback.time_predict_valid[i]], ["NaN", eval_callback.time_predict_test[i]]] for i in range(self.fidelities["fidelity_values"][-1])]

        # FIXME: we should store the transform time in the fidelities
        # train_transform_train is time_transform_train + time_transform_valid if early stopping is used otherwise time_transform_train
        # train_transform_predict is time_transform_train
        # valid_transform_train is "Nan"
        # valid_transform_predict is time_transform_valid
        # test_transform_train is "Nan"
        # test_transform_predict is time_transform_test

        # inner time between iterations
        inner_time = [eval_callback.time_after_iterations[i - 1] - eval_callback.time_before_iterations[i - 1] for i in eval_callback.iterations]
        print("Inner time:")
        print(sum(inner_time))

        # training time for all iterations
        # FIXME: if early stopping is used training time should also include the time to predict on the validation data and calculate the metric(s)
        inner_train_time = sum(eval_callback.time_train)
        print("Inner train time:")
        print(inner_train_time)

        # predict time (predict and metric calculation and storage) for all iterations
        inner_predict_train_time = sum(eval_callback.time_predict_train)
        inner_predict_valid_time = sum(eval_callback.time_predict_valid)
        inner_predict_test_time = sum(eval_callback.time_predict_test)
        print("Inner predict time:")
        print(inner_predict_train_time + inner_predict_valid_time + inner_predict_test_time)

        # overall overhead time (due to prepare and transform and all the calculated timings up until now)
        outer_overhead = time.time() - (time_after_fit - time_before_fit) - start_time
        print("Outer overall overhead:")
        print(outer_overhead)

        # overall fit time (including the overhead)
        fit_time = time.time() - start_time
        print("Overall fit time:")
        print(fit_time)

    def _predict(self, X):
        X_pred = self.pp_pipeline.transform(X)
        y_pred = self.learner.predict(X_pred)
        return self.encoder.inverse_transform(y_pred)

#if __name__ == "__main__":
#    # Just for quick testing
#    import numpy as np
#    import time
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
#    start_time = time.time()
#    workflow.fit(X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test, metadata=dataset_metadata)
#    stop_time = time.time() - start_time
#    print("Time:")
#    print(stop_time)
#    y_pred = workflow.predict(X_test)
#    print("Accuracy:")
#    print(accuracy_score(y_test, y_pred))

import logging
import traceback

import absl.logging
import numpy as np
import tensorflow as tf
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from lcdb.scorer import ClassificationScorer
from lcdb.timer import Timer
from lcdb.utils import get_iteration_schedule
from lcdb.workflow._base_workflow import BaseWorkflow
from lcdb.workflow.keras.utils import ACTIVATIONS, OPTIMIZERS, REGULARIZERS
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)

CONFIG_SPACE = ConfigurationSpace(
    name="keras._dense",
    space={
        "num_layers": Integer("num_layers", bounds=(1, 20), default=5),
        "num_units": Integer("num_units", bounds=(1, 200), log=True, default=32),
        "activation": Categorical("activation", items=ACTIVATIONS, default="relu"),
        "dropout_rate": Float("dropout_rate", bounds=(0.0, 0.9), default=0.1),
        "skip_co": Categorical("skip_co", items=[True, False], default=True),
        "optimizer": Categorical(
            "optimizer", items=list(OPTIMIZERS.keys()), default="Adam"
        ),
        "learning_rate": Float(
            "learning_rate", bounds=(1e-5, 10.0), log=True, default=1e-3
        ),
        "batch_size": Integer("batch_size", bounds=(1, 512), log=True, default=32),
        "num_epochs": Integer("num_epochs", bounds=(1, 100), log=True, default=10),
        "shuffle_each_epoch": Categorical(
            "shuffle_each_epoch", items=[True, False], default=True
        ),
        # TODO: add regularization hyperparameters
        "kernel_regularizer": Categorical(
            "kernel_regularizer", list(REGULARIZERS.keys()), default="none"
        ),
        "bias_regularizer": Categorical(
            "bias_regularizer", list(REGULARIZERS.keys()), default="none"
        ),
        "activity_regularizer": Categorical(
            "activity_regularizer", list(REGULARIZERS.keys()), default="none"
        ),
        "regularizer_factor": Float(
            "regularizer_factor", bounds=(0.0, 1.0), default=0.01
        ),
        # TODO: refine preprocessing
        "transform_real": Categorical(
            "transform_real", ["minmax", "std", "none"], default="none"
        ),
        "transform_cat": Categorical(
            "transform_cat", ["onehot", "ordinal"], default="onehot"
        ),
    },
)


class IterationCurveCallback(tf.keras.callbacks.Callback):
    def __init__(self, workflow: BaseWorkflow, timer: Timer, data: dict):
        super().__init__()
        self.timer = timer
        self.workflow = workflow
        self.data = data
        self.epoch = None
        self.scorer = ClassificationScorer(
            classes=self.workflow.infos["classes"], timer=self.timer
        )
        self.schedule = get_iteration_schedule(self.workflow.num_epochs)[::-1]

        # Safeguard to check timers
        self.train_timer_id = None
        self.test_timer_id = None
        self.epoch_timer_id = None

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs=logs)
        self.epoch = epoch
        self.epoch_timer_id = self.timer.start("epoch", metadata={"value": self.epoch})

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        assert self.timer.active_node.id == self.epoch_timer_id
        self.timer.stop()

    def on_train_begin(self, logs=None):
        self.train_timer_id = self.timer.start("epoch_train")

    def on_train_end(self, logs=None):
        assert self.timer.active_node.id == self.train_timer_id
        self.timer.stop()

    def on_test_begin(self, logs=None):
        # Manage the schedule
        epoch_schedule = self.schedule[-1]
        if self.epoch + 1 != epoch_schedule:
            return
        self.schedule.pop()

        with self.timer.time("epoch_test"):
            with self.timer.time("metrics"):
                for label_split, data_split in self.data.items():
                    with self.timer.time(label_split):
                        with self.timer.time("predict_with_proba"):
                            y_pred, y_pred_proba = self.workflow._predict_with_proba(
                                data_split["X"]
                            )

                        y_true = data_split["y"]

                        self.scorer.score(
                            y_true=y_true,
                            y_pred=y_pred,
                            y_pred_proba=y_pred_proba,
                        )


class DenseNNWorkflow(BaseWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(
        self,
        timer=None,
        num_layers=5,
        num_units=32,
        activation="relu",
        dropout_rate=0.1,
        skip_co=True,
        optimizer="Adam",
        learning_rate=0.001,
        batch_size=32,
        num_epochs=10,
        kernel_regularizer="none",
        bias_regularizer="none",
        activity_regularizer="none",
        regularizer_factor=0.01,
        shuffle_each_epoch=True,
        transform_real="none",
        transform_cat="onehot",
        verbose=0,
        **kwargs,
    ):
        super().__init__(timer)
        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = None if activation == "none" else activation
        self.dropout_rate = dropout_rate
        self.skip_co = skip_co
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_each_epoch = shuffle_each_epoch
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.regularizer_factor = regularizer_factor

        self.transform_real = transform_real
        self.transform_cat = transform_cat
        self._transformer_label = LabelEncoder()

        self.verbose = verbose

        tf.keras.backend.clear_session()

    @classmethod
    def config_space(cls):
        return cls._config_space

    def _transform(self, X, y, metadata):
        X_cat = X[:, metadata["categories"]["columns"]]
        X_real = X[:, ~metadata["categories"]["columns"]]

        has_cat = X_cat.shape[1] > 0
        has_real = X_real.shape[1] > 0

        if not (self.transform_fitted):
            # Categorical features
            if self.transform_cat == "onehot":
                self._transformer_cat = OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                )
            elif self.transform_cat == "ordinal":
                self._transformer_cat = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            else:
                raise ValueError(
                    f"Unknown categorical transformation {self.transform_cat}"
                )

            if has_cat:
                self._transformer_cat.fit(X_cat)

            # Real features
            if self.transform_real == "minmax":
                self._transformer_real = MinMaxScaler()
            elif self.transform_real == "std":
                self._transformer_real = StandardScaler()
            elif self.transform_real == "none":
                # No transformation
                self._transformer_real = FunctionTransformer(func=lambda x: x)
            else:
                raise ValueError(f"Unknown real transformation {self.transform_real}")

            if has_real:
                self._transformer_real.fit(X_real)

        if has_cat:
            X_cat = self._transformer_cat.transform(X_cat)
        if has_real:
            X_real = self._transformer_real.transform(X_real)
        X = np.concatenate([X_real, X_cat], axis=1)
        return X

    def build_model(self, input_shape, num_classes):
        inputs = out = tf.keras.Input(shape=input_shape)

        prev = None

        # Model layers
        for layer_i in range(self.num_layers):
            out = tf.keras.layers.Dense(
                self.num_units,
                activation=self.activation,
                activity_regularizer=REGULARIZERS[self.activity_regularizer](self.regularizer_factor),
                kernel_regularizer=REGULARIZERS[self.kernel_regularizer](self.regularizer_factor),
                bias_regularizer=REGULARIZERS[self.bias_regularizer](self.regularizer_factor),
            )(out)
            out = tf.keras.layers.Dropout(self.dropout_rate)(out)

            if self.skip_co and prev is not None:
                out = out + prev
            prev = out

        # Model output
        layer_logits = tf.keras.layers.Dense(
            self.num_units, activation=self.activation
        )(out)
        layer_proba = tf.keras.layers.Dense(num_classes, activation="softmax")(
            layer_logits
        )

        model = tf.keras.Model(inputs=inputs, outputs=layer_proba)
        print(model.summary())

        return model

    def _fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata
        X = self.transform(X, y, metadata).astype(np.float32)
        y_ = self._transformer_label.fit_transform(y)

        self.infos["classes"] = list(self._transformer_label.classes_)

        # TODO: adapt timings for validation and test data
        X_valid = self.transform(X_valid, y_valid, metadata).astype(np.float32)
        y_valid_ = self._transformer_label.transform(y_valid)

        # TODO: adapt timings
        X_test = self.transform(X_test, y_test, metadata).astype(np.float32)

        self.learner = self.build_model(X.shape[1:], metadata["num_classes"])

        optimizer = OPTIMIZERS[self.optimizer]()
        optimizer.learning_rate = self.learning_rate

        self.learner.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[],
        )

        iteration_curve_callback = IterationCurveCallback(
            workflow=self,
            timer=self.timer,
            data=dict(
                train=dict(X=X, y=y),
                valid=dict(X=X_valid, y=y_valid),
                test=dict(X=X_test, y=y_test),
            ),
        )

        fit_history = self.learner.fit(
            X,
            y_,
            batch_size=min(len(X), self.batch_size),
            epochs=self.num_epochs,
            shuffle=self.shuffle_each_epoch,
            validation_data=(X_valid, y_valid_),
            callbacks=[
                iteration_curve_callback,
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.ReduceLROnPlateau(),
                tf.keras.callbacks.EarlyStopping(patience=10),
            ],
            verbose=self.verbose,
        ).history

    def _predict(self, X):
        y_pred = self._predict_proba(X).argmax(axis=1)
        y_pred = self._transformer_label.inverse_transform(y_pred)
        return y_pred

    def _predict_proba(self, X):
        X = self.transform(X, y=None, metadata=self.metadata).astype(np.float32)
        y_pred = self.learner.predict(
            X, batch_size=min(len(X), self.batch_size), verbose=self.verbose
        )
        return y_pred

    def _predict_with_proba(self, X):
        y_pred_proba = self._predict_proba(X)
        y_pred = y_pred_proba.argmax(axis=1)
        y_pred = self._transformer_label.inverse_transform(y_pred)
        return y_pred, y_pred_proba

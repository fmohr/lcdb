import importlib
import logging
import time

import absl.logging
import numpy as np
import tensorflow as tf
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from .._base_workflow import BaseWorkflow
from .utils import ACTIVATIONS, OPTIMIZERS

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)

CONFIG_SPACE = ConfigurationSpace(
    name="keras._dense",
    space={
        "num_units": Integer("num_units", bounds=(1, 200), log=True, default=32),
        "activation": Categorical("activation", items=ACTIVATIONS, default="relu"),
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
        # TODO: refine preprocessing
        "transform_real": Categorical(
            "transform_real", ["minmax", "std", "none"], default="none"
        ),
        "transform_cat": Categorical(
            "transform_cat", ["onehot", "ordinal"], default="onehot"
        ),
    },
)


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.timestamp_on_train_begin = None
        self.times = []

    def on_train_begin(self, logs=None):
        self.timestamp_on_train_begin = time.time()

    def on_train_end(self, logs=None):
        times = np.asarray(self.times)
        times -= self.timestamp_on_train_begin
        self.times = times.tolist()

    def on_test_end(self, logs=None):
        self.times.append(time.time())


class DenseNNWorkflow(BaseWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE

    def __init__(
        self,
        num_units=32,
        activation="relu",
        optimizer="Adam",
        learning_rate=0.001,
        batch_size=32,
        num_epochs=10,
        shuffle_each_epoch=True,
        transform_real="none",
        transform_cat="onehot",
        verbose=0,
        **kwargs,
    ):
        super().__init__()
        self.requires_valid_to_fit = True

        self.num_units = num_units
        self.activation = None if activation == "none" else activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_each_epoch = shuffle_each_epoch

        self.transform_real = transform_real
        self.transform_cat = transform_cat
        self._transformer_label = LabelEncoder()

        self.verbose = verbose

        self.fidelities = {
            "fidelity_unit": "epochs",
            "fidelity_values": [],
            "score_types": ["loss", "accuracy", "val_loss", "val_accuracy"],
            "score_values": [],
            "time_types": ["epoch"],
            "time_values": [],
        }

    @classmethod
    def config_space(cls):
        # TODO: If the config_space needs to be expanded with preprocessing module it should be done here
        return cls._config_space

    def _transform(self, X, metadata):
        X_cat = X[:, metadata["categories"]["columns"]]
        X_real = X[:, ~metadata["categories"]["columns"]]

        has_cat = X_cat.shape[1] > 0
        has_real = X_real.shape[1] > 0

        if not (self.transform_fitted):
            # Categorical features
            if self.transform_cat == "onehot":
                self._transformer_cat = OneHotEncoder(drop="first", sparse_output=False)
            elif self.transform_cat == "ordinal":
                self._transformer_cat = OrdinalEncoder()
            else:
                raise ValueError(
                    f"Unknown categorical transformation {self.transform_cat}"
                )

            if metadata["categories"]["values"] is not None:
                max_categories = max(len(x) for x in metadata["categories"]["values"])
                values = np.array(
                    [
                        c_val + [c_val[-1]] * (max_categories - len(c_val))
                        for c_val in metadata["categories"]["values"]
                    ]
                ).T
            else:
                values = X_cat

            if has_cat:
                self._transformer_cat.fit(values)

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
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                self.num_units, input_shape=input_shape, activation=self.activation
            )
        )
        model.add(tf.keras.layers.Dense(self.num_units, activation=self.activation))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        return model

    def _fit(self, X, y, X_valid, y_valid, metadata):
        self.metadata = metadata
        X = self.transform(X, metadata).astype(np.float32)
        y = self._transformer_label.fit_transform(y)

        X_valid = self.transform(X_valid, metadata).astype(np.float32)
        y_valid = self._transformer_label.transform(y_valid)

        self.learner = self.build_model(X.shape[1:], metadata["num_classes"])

        optimizer = OPTIMIZERS[self.optimizer]()
        optimizer.learning_rate = self.learning_rate

        self.learner.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        timing_callback = TimingCallback()

        fit_history = self.learner.fit(
            X,
            y,
            batch_size=min(len(X), self.batch_size),
            epochs=self.num_epochs,
            shuffle=self.shuffle_each_epoch,
            validation_data=(X_valid, y_valid),
            callbacks=[
                timing_callback,
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.ReduceLROnPlateau(),
                tf.keras.callbacks.EarlyStopping(patience=10),
            ],
            verbose=self.verbose,
        ).history

        self.fidelities["fidelity_values"] = (
            np.arange(len(fit_history["loss"])) + 1
        ).tolist()
        self.fidelities["score_values"] = [
            list(scores)
            for scores in zip(*(fit_history[k] for k in self.fidelities["score_types"]))
        ]
        self.fidelities["time_values"] = timing_callback.times

    def _predict(self, X):
        X = self.transform(X, metadata=self.metadata).astype(np.float32)
        y_pred = self.learner.predict(
            X, batch_size=min(len(X), self.batch_size), verbose=self.verbose
        ).argmax(axis=1)
        y_pred = self._transformer_label.inverse_transform(y_pred)
        return y_pred

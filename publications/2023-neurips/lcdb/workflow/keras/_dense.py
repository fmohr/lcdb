import keras
import pandas as pd
import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from ...scorer import ClassificationScorer
from ...timer import Timer
from ...utils import get_schedule, filter_keys_with_prefix
from .._base_workflow import BaseWorkflow
from .._preprocessing_workflow import PreprocessedWorkflow
from ._augmentation import augment_with_mixup
from .utils import (
    ACTIVATIONS,
    INITIALIZERS,
    OPTIMIZERS,
    REGULARIZERS,
    count_params,
)

from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# regularization techniques
from ._lookahead import Lookahead
from ._swa import SWA
from ._snapshot import Snapshot

CONFIG_SPACE = ConfigurationSpace(
    name="keras._dense",
    space={
        "num_layers": Integer("num_layers", bounds=(1, 20), default=9),
        "num_units": Integer("num_units", bounds=(1, 4096), log=True, default=512),
        "activation": Categorical("activation", items=ACTIVATIONS, default="relu"),
        "dropout_rate": Float("dropout_rate", bounds=(0.0, 0.9), default=0.1),
        "skip_co": Categorical("skip_co", items=[True, False], default=True),
        "batch_norm": Categorical("batch_norm", items=[True, False], default=False),
        "optimizer": Categorical(
            "optimizer", items=list(OPTIMIZERS.keys()), default="SGD"
        ),
        "learning_rate": Float(
            "learning_rate", bounds=(1e-5, 10.0), log=True, default=1e-4
        ),
        "batch_size": Integer("batch_size", bounds=(1, 512), log=True, default=32),
        # "num_epochs": Integer("num_epochs", bounds=(1, 100), log=True, default=10),
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
        "kernel_initializer": Categorical(
            "kernel_initializer", INITIALIZERS, default="glorot_uniform"
        ),
        "stochastic_weight_averaging": Categorical(
            "stochastic_weight_averaging", items=[True, False], default=False
        ),
        "lookahead": Categorical(
            "lookahead", items=[True, False], default=False
        ),
        "snapshot_ensembles": Categorical(
            "snapshot_ensembles", items=[False, True], default=False
        ),
        # TODO: The following snapshot parameters should be made conditional and only appear if snapshot_ensembles=True
        "snapshot_ensembles_period": Integer(
            "snapshot_ensembles_period", bounds=(2, 100), default=20
        ),
        "snapshot_ensembles_reset_weights": Categorical(
            "snapshot_ensembles_reset_weights", items=[False, True], default=False
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


class IterationCurveCallback(keras.callbacks.Callback):
    def __init__(
        self,
        workflow: BaseWorkflow,
        timer: Timer,
        data: dict,
        epoch_schedule: str = "power",
    ):
        super().__init__()
        self.timer = timer
        self.workflow = workflow
        self.data = data
        self.epoch = None
        self.scorer = ClassificationScorer(
            classes_learner=self.workflow.infos["classes_train"],
            classes_overall=self.workflow.infos["classes_overall"],
            timer=self.timer
        )
        self.schedule = get_schedule(
            name=epoch_schedule, n=self.workflow.num_epochs, base=2, power=0.5, delay=0
        )[::-1]

        # Safeguard to check timers
        self.train_timer_id = None
        self.test_timer_id = None
        self.epoch_timer_id = None

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs=logs)
        self.epoch = epoch
        self.epoch_timer_id = self.timer.start(
            "epoch", metadata={"value": self.epoch + 1}
        )
        self.train_timer_id = self.timer.start("epoch_train")

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        assert self.timer.active_node.id == self.epoch_timer_id
        self.timer.stop()

    def on_test_begin(self, logs=None):
        assert self.timer.active_node.id == self.train_timer_id
        self.timer.stop()

        # Manage the schedule
        epoch_schedule = self.schedule[-1]
        is_epoch_to_test = (self.epoch + 1) == epoch_schedule
        is_training_continued = not (self.model.stop_training)
        if not (is_epoch_to_test) and is_training_continued:
            return
        self.schedule.pop()

        with self.timer.time("epoch_test"):
            with self.timer.time("metrics"):
                for label_split, data_split in self.data.items():
                    with self.timer.time(label_split):
                        with self.timer.time("predict_with_proba"):
                            (
                                y_pred,
                                y_pred_proba,
                            ) = self.workflow._predict_with_proba_after_transform(
                                data_split["X"],
                                use_snapshot_ensemble=True
                            )

                        y_true = data_split["y"]
                        scores = self.scorer.score(
                            y_true=y_true,
                            y_pred=y_pred,
                            y_pred_proba=y_pred_proba,
                        )
                        print(scores)


class DenseNNWorkflow(PreprocessedWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="pp",
        delimiter="@",
        configuration_space=PreprocessedWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        num_layers=5,
        num_units=32,
        activation="relu",
        dropout_rate=0.1,
        skip_co=True,
        batch_norm=False,
        optimizer="Adam",
        learning_rate=0.001,
        batch_size=32,
        num_epochs=1000,
        kernel_regularizer="none",
        bias_regularizer="none",
        activity_regularizer="none",
        regularizer_factor=0.01,
        kernel_initializer="glorot_uniform",
        stochastic_weight_averaging=False,
        lookahead=False,
        snapshot_ensembles=False,
        snapshot_ensembles_period=20,
        snapshot_ensembles_reset_weights=False,
        shuffle_each_epoch=True,
        transform_real="none",
        transform_cat="onehot",
        verbose=2,
        epoch_schedule: str = "power",
        **kwargs,
    ):
        """Dense Neural Network Workflow implements the class of multi-layer fully connected neural networks.

        Args:
            timer (_type_, optional): _description_. Defaults to None.
            num_layers (int, optional): _description_. Defaults to 5.
            num_units (int, optional): _description_. Defaults to 32.
            activation (str, optional): _description_. Defaults to "relu".
            dropout_rate (float, optional): _description_. Defaults to 0.1.
            skip_co (bool, optional): _description_. Defaults to True.
            batch_norm (bool, optional): _description_. Defaults to False.
            optimizer (str, optional): _description_. Defaults to "Adam".
            learning_rate (float, optional): _description_. Defaults to 0.001.
            batch_size (int, optional): _description_. Defaults to 32.
            num_epochs (int, optional): _description_. Defaults to 200.
            kernel_regularizer (str, optional): _description_. Defaults to "none".
            bias_regularizer (str, optional): _description_. Defaults to "none".
            activity_regularizer (str, optional): _description_. Defaults to "none".
            regularizer_factor (float, optional): _description_. Defaults to 0.01.
            kernel_initializer (str, optional): _description_. Defaults to "glorot_uniform".
            stochastic_weight_averaging (bool, optional):
            lookahead (bool, optional):
            shuffle_each_epoch (bool, optional): _description_. Defaults to True.
            transform_real (str, optional): _description_. Defaults to "none".
            transform_cat (str, optional): _description_. Defaults to "onehot".
            verbose (int, optional): _description_. Defaults to 0.
            epoch_schedule (str, optional): _description_. Defaults to "power".
        """

        super().__init__(timer, **filter_keys_with_prefix(kwargs, prefix="pp@"))
        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = None if activation == "none" else activation
        self.dropout_rate = dropout_rate
        self.skip_co = skip_co
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_each_epoch = shuffle_each_epoch
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.regularizer_factor = regularizer_factor
        self.kernel_initializer = kernel_initializer
        self.stochastic_weight_averaging = stochastic_weight_averaging
        self.lookahead = lookahead
        self.snapshot_ensembles = snapshot_ensembles
        self.snapshot_ensembles_period = snapshot_ensembles_period
        self.snapshot_ensembles_reset_weights = snapshot_ensembles_reset_weights
        self.snapshot_callback = None

        self.transform_real = transform_real
        self.transform_cat = transform_cat

        self.verbose = verbose
        self.epoch_schedule = epoch_schedule

        # state variables
        self.use_snapshot_models_for_prediction = False  # this variable is modified by the Snapshot callback

        keras.backend.clear_session()

    @classmethod
    def config_space(cls):
        return cls._config_space

    def build_model(self, input_shape, num_classes):
        inputs = out = keras.Input(shape=input_shape)

        prev = None

        # Model layers
        for layer_i in range(self.num_layers):
            out = keras.layers.Dense(
                self.num_units,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                activity_regularizer=REGULARIZERS[self.activity_regularizer](
                    self.regularizer_factor
                ),
                kernel_regularizer=REGULARIZERS[self.kernel_regularizer](
                    self.regularizer_factor
                ),
                bias_regularizer=REGULARIZERS[self.bias_regularizer](
                    self.regularizer_factor
                ),
            )(out)
            if self.batch_norm:
                out = keras.layers.BatchNormalization()(out)
            out = keras.layers.Dropout(self.dropout_rate)(out)

            if self.skip_co and prev is not None:
                out = out + prev
            prev = out

        # Model output
        layer_logits = keras.layers.Dense(self.num_units, activation=self.activation)(
            out
        )
        layer_proba = keras.layers.Dense(num_classes, activation="softmax")(
            layer_logits
        )

        model = keras.Model(inputs=inputs, outputs=layer_proba)

        return model

    def _encode_label_vector(self, y):
        return np.array([
            self.infos["classes_train"].index(self.infos["classes_overall"][label])
            for label in y
        ])

    def _decode_label_vector(self, y):
        return np.array([
            self.infos["classes_overall"].index(self.infos["classes_train"][i])
            for i in y
        ])

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata

        # create internal labels for keras ordered from 0 to k-1 where, k is the number of labels *known* to the NN
        mask_valid = np.isin(y_valid, self.infos["classes_train"])
        self.y_valid_label_encoded_keras = np.array([
            self.infos["classes_train"].index(label) for label in y_valid[mask_valid]
        ])

        # build skeleton of neural network
        self.learner = self.build_model(X.shape[1:], len(self.infos["classes_train"]))

        # Count Parameters in Model and Record
        if self.timer.root.metadata.get("num_parameters_train") is None:
            params = count_params(self.learner)
            self.timer.root["num_parameters_not_train"] = params[
                "num_parameters_not_train"
            ]
            self.timer.root["num_parameters_train"] = params["num_parameters_train"]

        optimizer = OPTIMIZERS[self.optimizer]()
        optimizer.learning_rate = self.learning_rate

        iteration_curve_callback = IterationCurveCallback(
            workflow=self,
            timer=self.timer,
            data=dict(
                train=dict(X=X, y=y),
                val=dict(X=X_valid, y=y_valid),
                test=dict(X=X_test, y=y_test),
            ),
            epoch_schedule=self.epoch_schedule,
        )

        # define callbacks
        callbacks = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ReduceLROnPlateau(),
            keras.callbacks.EarlyStopping(patience=100),
            iteration_curve_callback,
        ]

        base_optimizer = optimizer
        if self.lookahead:
            optimizer = Lookahead(optimizer, learning_rate=0.5, la_steps=10)

        if self.stochastic_weight_averaging:
            callbacks.append(SWA(start_epoch=2))

        if self.snapshot_ensembles:
            self.snapshot_callback = Snapshot(
                workflow=self,
                optimizer=base_optimizer,
                reset_weights=self.snapshot_ensembles_reset_weights,
                period=self.snapshot_ensembles_period
            )
            callbacks.append(self.snapshot_callback)

        self.learner.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # now fit model
        self.learner.fit(
            X,
            self._encode_label_vector(y),
            batch_size=min(len(X), self.batch_size),
            epochs=self.num_epochs,
            shuffle=self.shuffle_each_epoch,
            validation_data=(X_valid[mask_valid], self._encode_label_vector(y_valid[mask_valid])),
            callbacks=callbacks,
            verbose=self.verbose,
        )

    def _predict_after_transform(self, X):
        return self._predict_with_proba_after_transform(X)[0]

    def _predict_proba_after_transform(self, X, use_snapshot_ensemble=False):
        assert not np.any(pd.isna(X)), "Found NANs in the input (after transformation) for the NN prediction!"

        # determine models used to make prediction (usually just the model itself unless snapshot ensembles are used)
        models = [self.learner]
        if use_snapshot_ensemble and self.snapshot_callback is not None:
            models.extend(self.snapshot_callback.checkpoint_models)

        # compute probabilities per model
        y_pred_proba = []
        for model in models:
            y_pred_proba_model = model.predict(
                X, batch_size=min(len(X), self.batch_size), verbose=self.verbose
            )
            y_pred_proba.append(y_pred_proba_model)

        # average probabilities
        y_pred_proba = np.array(y_pred_proba)
        assert not np.any(np.isnan(y_pred_proba)), "There are NAN values in the NN prediction!"
        return y_pred_proba.mean(axis=0)

    def _predict_with_proba_after_transform(self, X, use_snapshot_ensemble=False):

        # obtain probabilistic prediction from snapshot ensemble
        y_pred_proba = self._predict_proba_after_transform(X, use_snapshot_ensemble=use_snapshot_ensemble)

        # derive predictions
        y_pred = self._decode_label_vector(y_pred_proba.argmax(axis=1))

        return y_pred, y_pred_proba

import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from ...scorer import ClassificationScorer
from ...timer import Timer
from ...utils import get_schedule, filter_keys_with_prefix
from .._base_workflow import BaseWorkflow
from .._preprocessing_workflow import PreprocessedWorkflow
from ._augmentation import MixUpAugmentation, CutMixAugmentation, CutOutAugmentation
from .utils import (
    ACTIVATIONS,
    INITIALIZERS,
    OPTIMIZERS,
    REGULARIZERS,
    count_params,
)

from keras.src.backend import convert_to_numpy

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
        "shuffle_each_epoch": Categorical(
            "shuffle_each_epoch", items=[True, False], default=True
        ),
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
        "lookahead_learning_rate": Float(
            "lookahead_learning_rate", bounds=(0.2, 0.8), default=0.2
        ),
        "lookahead_num_steps": Integer(
            "lookahead_num_steps", bounds=(2, 10), default=5
        ),
        "snapshot_ensemble": Categorical(
            "snapshot_ensemble", items=[False, True], default=False
        ),
        # TODO: The following snapshot parameters should be made conditional and only appear if snapshot_ensembles=True
        "snapshot_ensemble_period_init": Integer(
            "snapshot_ensemble_period_init", bounds=(2, 100), default=20
        ),
        "snapshot_ensemble_period_increase": Integer(
            "snapshot_ensemble_period_increase", bounds=(0, 5), default=0
        ),
        "snapshot_ensemble_reset_weights": Categorical(
            "snapshot_ensemble_reset_weights", items=[False, True], default=False
        ),
        "data_augmentation": Categorical(
            "data_augmentation", items=["none", "cutout", "mixup", "cutmix"], default="none"
        ),
        # TODO: The following data augmentation parameters should be made conditional and only appear if snapshot_ensembles=True
        "data_augmentation_cutout_patch_ratio": Float(
            "data_augmentation_cutout_patch_ratio", bounds=(0.0, 1.0), default=0.1
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
            "epoch",
            metadata={
                "epoch_cnt": self.epoch + 1,
                "learning_rate": round(float(convert_to_numpy(self.workflow.learner.optimizer._learning_rate)), 8)
            })
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

        with self.timer.time("epoch_test") as timer:
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


class AugmentDataGenerator(Sequence):
    def __init__(self, X, y, batch_size, augmenters, encode_label_vector, shuffle=True):
        self.X = tf.convert_to_tensor(X)
        self.y = tf.convert_to_tensor(encode_label_vector(y))
        self.batch_size = batch_size
        self.augmenters = augmenters
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = tf.gather(self.X, batch_indices)
        y_batch = tf.gather(self.y, batch_indices)

        for augmenter in self.augmenters:
            X_batch, y_batch = augmenter.augment(X_batch, y_batch)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)



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
        num_epochs=100,
        num_epochs_patience=100,
        kernel_regularizer="none",
        bias_regularizer="none",
        activity_regularizer="none",
        regularizer_factor=0.01,
        kernel_initializer="glorot_uniform",
        stochastic_weight_averaging=False,
        lookahead=False,
        lookahead_learning_rate: float = 0.5,
        lookahead_num_steps: int = 5,
        snapshot_ensemble=False,
        snapshot_ensemble_period_init=20,
        snapshot_ensemble_period_increase=0,
        snapshot_ensemble_reset_weights=False,
        data_augmentation="none",
        data_augmentation_cutout_patch_ratio: float = 0.1,
        shuffle_each_epoch=True,
        verbose=2,
        epoch_schedule: str = "full",
        random_state=None,
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

        # check kwargs
        for k in kwargs.keys():
            if not k.startswith("pp@"):
                raise ValueError(f"Unsupported hyperparameter for DenseNNWorkflow: {k} (with value ({kwargs[k]}).")

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
        self.num_epochs_patience = num_epochs_patience
        self.shuffle_each_epoch = shuffle_each_epoch
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.regularizer_factor = regularizer_factor
        self.kernel_initializer = kernel_initializer
        self.stochastic_weight_averaging = stochastic_weight_averaging
        self.lookahead = lookahead
        self.lookahead_learning_rate = lookahead_learning_rate
        self.lookahead_num_steps = lookahead_num_steps

        self.snapshot_ensemble = snapshot_ensemble
        self.snapshot_ensemble_period_init = snapshot_ensemble_period_init
        self.snapshot_ensemble_period_increase = snapshot_ensemble_period_increase
        self.snapshot_ensemble_reset_weights = snapshot_ensemble_reset_weights
        self.snapshot_callback = None

        self.data_augmentation = None if data_augmentation == "none" else data_augmentation
        self.data_augmentation_cutout_patch_ratio = data_augmentation_cutout_patch_ratio

        self.verbose = verbose
        self.epoch_schedule = epoch_schedule

        # state variables
        self.use_snapshot_models_for_prediction = False  # this variable is modified by the Snapshot callback
        self.random_state = random_state

        keras.backend.clear_session()

    @classmethod
    def config_space(cls):
        return cls._config_space

    def build_model(self, input_shape, num_classes):
        inputs = out = keras.Input(shape=input_shape)

        print(f"Building model with input layer size {input_shape}")

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
        int_encoded_labels = np.array([
            self.infos["classes_train"].index(self.infos["classes_overall"][label])
            for label in y
        ])
        one_hot_encoded_labels = keras.utils.to_categorical(
            int_encoded_labels,
            num_classes=len(self.infos["classes_train"])
        )
        return one_hot_encoded_labels

    def _decode_label_vector(self, y):
        out = np.array([
            self.infos["classes_overall"].index(self.infos["classes_train"][i])
            for i in y
        ])
        return out

    # def _transform_train_data_prior_to_standard_preprocessing(self, X, y):

    #     # apply data augmentation to the one-hot encoded training data
    #     y_one_hot = self._encode_label_vector(y)
    #     data_augmenters = []
    #     if self.data_augmentation == "cutout":
    #         data_augmenters.append(CutOutAugmentation(
    #             probability_of_cut=self.data_augmentation_cutout_patch_ratio,
    #             random_state=self.random_state
    #         ))
    #     elif self.data_augmentation == "mixup":
    #         data_augmenters.append(MixUpAugmentation(random_state=self.random_state))
    #     elif self.data_augmentation == "cutmix":
    #         data_augmenters.append(CutMixAugmentation(random_state=self.random_state))

    #     # apply data augmenter(s). In fact there can only be one currently, but we keep the code generic
    #     for data_augmenter in data_augmenters:
    #         X, y_one_hot = data_augmenter.augment(X, y_one_hot)

    #     return X, y_one_hot

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata

        # create internal labels for keras ordered from 0 to k-1 where, k is the number of labels *known* to the NN
        mask_valid = np.isin(y_valid, self.infos["classes_train"])

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
                # train=dict(X=X, y=np.argmax(y, axis=1)),  # assign the class with the highest true probability (1 except for if data augmentation is used)
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
            keras.callbacks.EarlyStopping(patience=self.num_epochs_patience),
        ]

        base_optimizer = optimizer
        if self.lookahead:
            optimizer = Lookahead(
                optimizer,
                learning_rate=self.lookahead_learning_rate,
                la_steps=self.lookahead_num_steps
            )

        if self.stochastic_weight_averaging:
            callbacks.append(SWA(start_epoch=2, batch_size=self.batch_size))

        if self.snapshot_ensemble:
            self.snapshot_callback = Snapshot(
                workflow=self,
                optimizer=base_optimizer,
                reset_weights=self.snapshot_ensemble_reset_weights,
                period_init=self.snapshot_ensemble_period_init,
                period_increase=self.snapshot_ensemble_period_increase
            )
            callbacks.append(self.snapshot_callback)

        # the callback for the iteration curve should be the last one
        callbacks.append(iteration_curve_callback)

        self.learner.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Prepare data augmenters
        data_augmenters = []
        if self.data_augmentation == "cutout":
            data_augmenters.append(CutOutAugmentation(
                probability_of_cut=self.data_augmentation_cutout_patch_ratio,
                random_state=self.random_state
            ))
        elif self.data_augmentation == "mixup":
            data_augmenters.append(MixUpAugmentation(random_state=self.random_state))
        elif self.data_augmentation == "cutmix":
            data_augmenters.append(CutMixAugmentation(random_state=self.random_state))

        # data generator for augmentation
        train_generator = AugmentDataGenerator(
                            X, y, batch_size=self.batch_size, 
                            augmenters=data_augmenters,
                            encode_label_vector=self._encode_label_vector,
                            shuffle=self.shuffle_each_epoch
                        )
        # now fit model
        self.learner.fit(
            train_generator,
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
        assert not np.any(np.isnan(y_pred_proba)), f"There are NAN values in the NN prediction!\n{y_pred_proba}. Input was:\n{X}"
        return y_pred_proba.mean(axis=0)

    def _predict_with_proba_after_transform(self, X, use_snapshot_ensemble=False):

        # obtain probabilistic prediction from snapshot ensemble
        y_pred_proba = self._predict_proba_after_transform(X, use_snapshot_ensemble=use_snapshot_ensemble)

        # derive predictions
        y_pred = self._decode_label_vector(y_pred_proba.argmax(axis=1))

        return y_pred, y_pred_proba

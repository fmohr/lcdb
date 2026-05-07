import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import Sequence

from tensorflow.keras.initializers import get
from keras.layers import Activation
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer, EqualsCondition, InCondition
from lcdb.builder.scorer import ClassificationScorer
from lcdb.builder.timer import Timer
from lcdb.builder.utils import get_schedule, filter_keys_with_prefix
from lcdb.workflow._base_workflow import BaseWorkflow
from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from .utils import (
    ACTIVATIONS,
    INITIALIZERS,
    OPTIMIZERS,
    LRSCHEDULERS,
    REGULARIZERS,
    count_params,
)

from keras.src.backend import convert_to_numpy

# regularization techniques
from lcdb.workflow.keras._lookahead import Lookahead
from lcdb.workflow.keras._weight_averaging import WA
from lcdb.workflow.keras._snapshot import Snapshot

CONFIG_SPACE = ConfigurationSpace(
    name="keras._dense",
    space={
        "num_layers": Integer("num_layers", bounds=(1, 20), default=9),
        "num_units_first": Integer("num_units_first", bounds=(1, 4096), log=True, default=512),
        "num_units_last": Integer("num_units_last", bounds=(1, 4096), log=True, default=512),
        "activation": Categorical("activation", items=ACTIVATIONS, default="relu"),
        "dropout_rate": Float("dropout_rate", bounds=(0.0, 0.9), default=0.1),
        "skip_co": Categorical("skip_co", items=[True, False], default=True),
        "batch_norm": Categorical("batch_norm", items=["off", "before_activation", "after_activation"], default="off"),
        "optimizer": Categorical("optimizer", items=list(OPTIMIZERS.keys()), default="SGD"),
        "learning_rate": Float("learning_rate", bounds=(1e-6, 10.0), log=True, default=1e-4),
        "anti_momentum_rate": Float("anti_momentum_rate", bounds=(1e-4, 0.2), log=True, default=1e-2),
        "anti_grad_var_rate": Float("anti_grad_var_rate", bounds=(1e-5, 0.2), log=True, default=1e-4),
        "learning_rate_scheduler": Categorical("learning_rate_scheduler", items=list(LRSCHEDULERS.keys())),
        "learning_rate_scheduler_decay_steps": Integer("learning_rate_scheduler_decay_steps", bounds=(1, 10**6), default=10**3, log=True),
        "learning_rate_scheduler_decay_rate": Float("learning_rate_scheduler_decay_rate", bounds=(0.8, 0.99), default=0.9),
        "learning_rate_scheduler_polynomial_end_learning_rate": Float("learning_rate_scheduler_polynomial_end_learning_rate", bounds=(1e-6, 1e-3), default=1e-5, log=True),
        "learning_rate_scheduler_polynomial_power": Float("learning_rate_scheduler_polynomial_power", bounds=(0.3, 3), default=1),
        "learning_rate_scheduler_cosine_alpha": Float("learning_rate_scheduler_cosine_alpha", bounds=(0, 1), default=0.0),
        "learning_rate_scheduler_cosine_restarts_tmul": Float("learning_rate_scheduler_cosine_restarts_tmul", bounds=(0.1, 3), default=2.0),
        "learning_rate_scheduler_cosine_restarts_mmul": Float("learning_rate_scheduler_cosine_restarts_mmul", bounds=(0.5, 2), default=1.0),
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
        "weight_averaging": Categorical(
            "weight_averaging", items=[True, False], default=False
        ),
        "weight_averaging_cycle_length": Integer(
            "weight_averaging_cycle_length", bounds=(1, 100), default=5
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
        "shake_shake": Categorical(
            "shake_shake", items=[False, True], default=False
        ),
        "shake_drop": Categorical( # requires skip connection
            "shake_drop", items=[False, True], default=False
        ),
        "shake_drop_drop_proba": Float(
            "shake_drop_drop_proba", bounds=(0.1, 0.9), default=0.5
        ),
        "data_augmentation": Categorical(
            "data_augmentation", items=["none", "cutout", "mixup", "cutmix"], default="none"
        ),
        "data_augmentation_cutout_patch_ratio": Float(
            "data_augmentation_cutout_patch_ratio", bounds=(0.0, 1.0), default=0.1
        ),
    },
)

CONFIG_SPACE.add([
        
    # enable shake drop only if there are skip connections
    EqualsCondition(CONFIG_SPACE["shake_drop"], CONFIG_SPACE["skip_co"], True),
    EqualsCondition(CONFIG_SPACE["shake_drop_drop_proba"], CONFIG_SPACE["shake_drop"], True),

    # optimizers
    InCondition(CONFIG_SPACE["anti_momentum_rate"], CONFIG_SPACE["optimizer"], ["SGD", "Adam", "AdamW", "Adamax", "Nadam", "RMSprop"]),
    InCondition(CONFIG_SPACE["anti_grad_var_rate"], CONFIG_SPACE["optimizer"], ["Adam", "AdamW", "Adamax", "Nadam", "RMSprop", "Adadelta"]),

    # learning rate schedulers
    InCondition(CONFIG_SPACE["learning_rate_scheduler_decay_steps"], CONFIG_SPACE["learning_rate_scheduler"], ["ReduceLROnPlateau", "ExponentialDecay", "PolynomialDecay", "InverseTimeDecay", "CosineDecay", "CosineDecayRestarts"]),
    InCondition(CONFIG_SPACE["learning_rate_scheduler_decay_rate"], CONFIG_SPACE["learning_rate_scheduler"], ["ReduceLROnPlateau", "ExponentialDecay", "InverseTimeDecay"]),
    InCondition(CONFIG_SPACE["learning_rate_scheduler_polynomial_end_learning_rate"], CONFIG_SPACE["learning_rate_scheduler"], ["PolynomialDecay"]),
    InCondition(CONFIG_SPACE["learning_rate_scheduler_cosine_alpha"], CONFIG_SPACE["learning_rate_scheduler"], ["CosineDecay", "CosineDecayRestarts"]),
    InCondition(CONFIG_SPACE["learning_rate_scheduler_cosine_restarts_tmul"], CONFIG_SPACE["learning_rate_scheduler"], ["CosineDecayRestarts"]),
    InCondition(CONFIG_SPACE["learning_rate_scheduler_cosine_restarts_mmul"], CONFIG_SPACE["learning_rate_scheduler"], ["CosineDecayRestarts"]),

    # cycle length for weight averaging only if weight averaging is active
    EqualsCondition(CONFIG_SPACE["weight_averaging_cycle_length"], CONFIG_SPACE["weight_averaging"], True),

    # configure snapshot ensemble increase only if the flag is on
    EqualsCondition(CONFIG_SPACE["snapshot_ensemble_period_init"], CONFIG_SPACE["snapshot_ensemble"], True),
    EqualsCondition(CONFIG_SPACE["snapshot_ensemble_period_increase"], CONFIG_SPACE["snapshot_ensemble"], True),
    EqualsCondition(CONFIG_SPACE["snapshot_ensemble_reset_weights"], CONFIG_SPACE["snapshot_ensemble"], True),

    # configure data augmentation patch ratio only if that augmentation is active
    EqualsCondition(CONFIG_SPACE["data_augmentation_cutout_patch_ratio"], CONFIG_SPACE["data_augmentation"], "cutout")
])


class IterationCurveCallback(keras.callbacks.Callback):
    def __init__(
        self,
        workflow: BaseWorkflow,
        timer: Timer,
        data: dict,
        logger,
        epoch_schedule: str = "power",
    ):
        super().__init__()
        self.timer = timer
        self.workflow = workflow
        self.data = data
        self.epoch = None
        self.logger = logger
        self.scorer = ClassificationScorer(
            classes_learner=self.workflow.infos["classes_train"],
            classes_overall=self.workflow.infos["classes_overall"],
            timer=self.timer
        )
        self.schedule = get_schedule(
            name=epoch_schedule, max_anchor=self.workflow.num_epochs, base=2, power=0.5, delay=0
        )
        if len(self.schedule) == 0:
            raise ValueError(f"Generated an empty schedule.")
        self.logger.info(f"Epoch schedule set to {self.schedule}")
        self.schedule = self.schedule[::-1]

        # Safeguard to check timers
        self.train_timer_id = None
        self.test_timer_id = None
        self.epoch_timer_id = None

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs=logs)
        self.epoch = epoch
        cur_learning_rate = round(float(convert_to_numpy(self.workflow.learner.optimizer.learning_rate)), 8)
        self.epoch_timer_id = self.timer.start(
            "epoch",
            metadata={
                "epoch_cnt": self.epoch + 1,
                "learning_rate": cur_learning_rate
            })
        self.train_timer_id = self.timer.start("epoch_train")
        self.logger.info(f"Starting epoch {epoch + 1}. Current learning rate is {cur_learning_rate}")

    def on_epoch_end(self, epoch, logs=None):        
        super().on_epoch_end(epoch, logs)
        assert self.timer.active_node.id == self.epoch_timer_id
        self.timer.stop()
        self.logger.info(f"Finished epoch {epoch + 1}")

    def on_test_begin(self, logs=None):
        assert self.timer.active_node.id == self.train_timer_id
        self.timer.stop()

        # Manage the schedule
        if not self.schedule:
            return
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
                        self.logger.info(f"Scores for {label_split} split are {scores}")


class AugmentDataGenerator(Sequence):
    def __init__(self, X, y, batch_size, augmenters, encode_label_vector, shuffle=True, random_state=None):
        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.y = tf.convert_to_tensor(encode_label_vector(y), dtype=tf.float32)
        self.batch_size = batch_size
        self.augmenters = augmenters
        self.shuffle = shuffle
        self.random_state = random_state if random_state is not None else np.random.RandomState()
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
            self.random_state.shuffle(self.indices)



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
        num_units_first=512,
        num_units_last=512,
        activation="relu",
        dropout_rate=0.1,
        skip_co=True,
        batch_norm="off",
        optimizer="Adam",
        learning_rate=0.001,
        anti_momentum_rate=0.1,
        anti_grad_var_rate=0.001,
        learning_rate_scheduler="none",
        learning_rate_scheduler_decay_steps=10**3,
        learning_rate_scheduler_decay_rate=0.9,
        learning_rate_scheduler_polynomial_end_learning_rate=1e-5,
        learning_rate_scheduler_polynomial_power=1,
        learning_rate_scheduler_cosine_alpha=0.0,
        learning_rate_scheduler_cosine_restarts_tmul=2.0,
        learning_rate_scheduler_cosine_restarts_mmul=1.0,
        batch_size=32,
        num_epochs=2048,
        num_epochs_patience=20,
        kernel_regularizer="none",
        bias_regularizer="none",
        activity_regularizer="none",
        regularizer_factor=0.01,
        kernel_initializer="glorot_uniform",
        weight_averaging=False,
        weight_averaging_cycle_length=5,
        lookahead=False,
        lookahead_learning_rate: float = 0.5,
        lookahead_num_steps: int = 5,
        snapshot_ensemble=False,
        snapshot_ensemble_period_init=20,
        snapshot_ensemble_period_increase=0,
        snapshot_ensemble_reset_weights=False,
        shake_shake=False,
        shake_drop=False,
        shake_drop_drop_proba=0.5,
        data_augmentation="none",
        data_augmentation_cutout_patch_ratio: float = 0.1,
        shuffle_each_epoch=True,
        epoch_schedule: str = "full",
        random_state=None,
        logger=None,
        raise_exception_on_unsuitable_preprocessor=True,
        memory_limit_in_bytes=None,
        n_jobs=None, # this will be ignored since this workflow is not parallelizable
        **kwargs,
    ):

        # check kwargs
        for k in kwargs.keys():
            if not k.startswith("pp@"):
                raise ValueError(f"Unsupported hyperparameter for DenseNNWorkflow: {k} (with value ({kwargs[k]}).")

        super().__init__(
            timer=timer,
            logger=logger,
            random_state=random_state,
            raise_exception_on_unsuitable_preprocessor=raise_exception_on_unsuitable_preprocessor,
            memory_limit_in_bytes=memory_limit_in_bytes,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )
        self.requires_valid_to_fit = True
        self.requires_test_to_fit = True

        self.num_layers = num_layers
        self.num_units_first = num_units_first
        self.num_units_last = num_units_last
        self.activation = None if activation == "none" else activation
        self.dropout_rate = dropout_rate
        self.skip_co = skip_co
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.anti_momentum_rate = anti_momentum_rate
        self.anti_grad_var_rate = anti_grad_var_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_decay_steps = learning_rate_scheduler_decay_steps
        self.learning_rate_scheduler_decay_rate = learning_rate_scheduler_decay_rate
        self.learning_rate_scheduler_polynomial_end_learning_rate = learning_rate_scheduler_polynomial_end_learning_rate
        self.learning_rate_scheduler_polynomial_power = learning_rate_scheduler_polynomial_power
        self.learning_rate_scheduler_cosine_alpha = learning_rate_scheduler_cosine_alpha
        self.learning_rate_scheduler_cosine_restarts_tmul = learning_rate_scheduler_cosine_restarts_tmul
        self.learning_rate_scheduler_cosine_restarts_mmul = learning_rate_scheduler_cosine_restarts_mmul
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_epochs_patience = num_epochs_patience
        self.shuffle_each_epoch = shuffle_each_epoch
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.regularizer_factor = regularizer_factor
        self.kernel_initializer = kernel_initializer
        self.weight_averaging = weight_averaging
        self.weight_averaging_cycle_length = weight_averaging_cycle_length
        self.lookahead = lookahead
        self.lookahead_learning_rate = lookahead_learning_rate
        self.lookahead_num_steps = lookahead_num_steps

        self.snapshot_ensemble = snapshot_ensemble
        self.snapshot_ensemble_period_init = snapshot_ensemble_period_init
        self.snapshot_ensemble_period_increase = snapshot_ensemble_period_increase
        self.snapshot_ensemble_reset_weights = snapshot_ensemble_reset_weights
        self.snapshot_callback = None

        self.shake_shake = shake_shake
        self.shake_drop = shake_drop
        if shake_drop and not skip_co:
            raise ValueError(f"Cannot have shake_drop enabled without skip connections enabled.")
        self.shake_drop_drop_proba = shake_drop_drop_proba

        self.data_augmentation = None if data_augmentation == "none" else data_augmentation
        self.data_augmentation_cutout_patch_ratio = data_augmentation_cutout_patch_ratio

        self.epoch_schedule = epoch_schedule

        # state variables
        self.use_snapshot_models_for_prediction = False  # this variable is modified by the Snapshot callback

        keras.backend.clear_session()

    @classmethod
    def config_space(cls):
        return cls._config_space

    @classmethod
    def builds_iteration_curve(cls):
        return True
    
    def build_model(self, input_shape, num_classes):

        inputs = out = keras.Input(shape=input_shape)

        self.logger.info(
            "Building model with the following config: "
            f"\n\tInput layer size: {input_shape}"
            f"\n\tMax number of Epochs: {self.num_epochs} with schedule for iteration curve: {self.epoch_schedule}"
        )
        if len(tf.config.list_physical_devices('GPU')) == 0:
            self.logger.warning("No GPU available or used. Training will be slow!")

        prev = None

        def _get_seeded_kernel_initializer():
            initializer = get(self.kernel_initializer)
            kwargs_initializer = {}
            if self.kernel_initializer not in ["ones"]:
                kwargs_initializer["seed"] = self.random_state.randint(0, 10**5)
            return initializer.__class__(**kwargs_initializer)

        def _build_block(_out, num_units):

            # Get initializer class
            seeded_initializer = _get_seeded_kernel_initializer()

            _out = keras.layers.Dense(
                num_units,
                activation=self.activation if self.batch_norm in ["off", "after_activation"] else None,
                kernel_initializer=seeded_initializer,
                activity_regularizer=REGULARIZERS[self.activity_regularizer](
                    self.regularizer_factor
                ),
                kernel_regularizer=REGULARIZERS[self.kernel_regularizer](
                    self.regularizer_factor
                ),
                bias_regularizer=REGULARIZERS[self.bias_regularizer](
                    self.regularizer_factor
                ),
            )(_out)
            if self.batch_norm != "off":
                _out = keras.layers.BatchNormalization()(_out)

                if self.batch_norm == "before_activation":
                    _out = Activation(self.activation)(_out)

            return keras.layers.Dropout(self.dropout_rate, seed=self.random_state.randint(0, 10**5))(_out)

        # Model layers
        num_units_in_layers = [int(n) for n in np.linspace(self.num_units_first, self.num_units_last, self.num_layers)]
        for layer_i, num_units in enumerate(num_units_in_layers):
            
            # create branch output (either standard, or shake-shaked)
            if not self.shake_shake:
                out = _build_block(out, num_units)
            else:
                from lcdb.workflow.keras._shake import ShakeShake
                out_1 = _build_block(out, num_units)
                out_2 = _build_block(out, num_units)
                out = ShakeShake(seed=self.random_state.randint(0, 10**5))([out_1, out_2])
            
            # apply shake drop if enabled
            if self.shake_drop:
                from lcdb.workflow.keras._shake import ShakeDrop
                out = ShakeDrop(
                    seed=self.random_state.randint(0, 10**5),
                    p_drop=self.shake_drop_drop_proba
                )(out)

            # add skip connection if enabled
            if self.skip_co and prev is not None:
                if self.num_units_first == self.num_units_last:
                    out = out + prev
                else:
                    out = out + keras.layers.Dense(
                        num_units,
                        kernel_initializer=_get_seeded_kernel_initializer()
                    )(prev)
            prev = out

        # Model output
        layer_logits = keras.layers.Dense(
            self.num_units_last,
            kernel_initializer=_get_seeded_kernel_initializer(),
            activation=self.activation)(
            out
        )
        layer_proba = keras.layers.Dense(
            num_classes, 
            kernel_initializer=_get_seeded_kernel_initializer(),
            activation="softmax")(
            layer_logits
        )

        return keras.Model(inputs=inputs, outputs=layer_proba)

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

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        self.metadata = metadata

        # create internal labels for keras ordered from 0 to k-1 where, k is the number of labels *known* to the NN
        mask_valid = np.isin(y_valid, self.infos["classes_train"])

        # move to the device (GPU if available, otherwise CPU)
        gpus = tf.config.list_logical_devices('GPU')

        if gpus:
            device_name = gpus[0].name
        else:
            cpus = tf.config.list_logical_devices('CPU')

            if not cpus:
                raise RuntimeError("No CPU device found")

            device_name = cpus[0].name
        with tf.device(device_name):

            self.logger.info(
                f"Preparing model construction. Hardware environment is as follows."
                f"\n\tDevices: {tf.config.list_physical_devices()}"
                f"\n\tNumber of GPUs available: {len(tf.config.list_physical_devices('GPU'))}"
                f"\n\tCHOSEN DEVICE (model is built and fit is invoked here in this context): {device_name}"
            )

            # build skeleton of neural network
            self.learner = self.build_model(X.shape[1:], len(self.infos["classes_train"]))

            # Count Parameters in Model and Record
            if self.timer.root.metadata.get("num_parameters_train") is None:
                params = count_params(self.learner)
                self.timer.root["num_parameters_not_train"] = params[
                    "num_parameters_not_train"
                ]
                self.timer.root["num_parameters_train"] = params["num_parameters_train"]

            # configure learning rate schedule
            if self.learning_rate_scheduler == "none":
                learning_rate_scheduler = self.learning_rate
            else:
                lrskwargs = {}
                if self.learning_rate_scheduler == "ReduceLROnPlateau":
                    lrskwargs["factor"] = self.learning_rate_scheduler_decay_rate
                    lrskwargs["patience"] = int(self.learning_rate_scheduler_decay_steps)
                elif self.learning_rate_scheduler == "ExponentialDecay":
                    lrskwargs["initial_learning_rate"] = self.learning_rate
                    lrskwargs["decay_rate"] = self.learning_rate_scheduler_decay_rate
                    lrskwargs["decay_steps"] = int(self.learning_rate_scheduler_decay_steps)
                elif self.learning_rate_scheduler == "PolynomialDecay":
                    lrskwargs["initial_learning_rate"] = self.learning_rate
                    lrskwargs["end_learning_rate"] = self.learning_rate_scheduler_polynomial_end_learning_rate
                    lrskwargs["power"] = self.learning_rate_scheduler_polynomial_power
                    lrskwargs["decay_steps"] = int(self.learning_rate_scheduler_decay_steps)
                elif self.learning_rate_scheduler == "InverseTimeDecay":
                    lrskwargs["initial_learning_rate"] = self.learning_rate
                    lrskwargs["decay_rate"] = self.learning_rate_scheduler_decay_rate
                    lrskwargs["decay_steps"] = int(self.learning_rate_scheduler_decay_steps)
                elif self.learning_rate_scheduler == "CosineDecay":
                    lrskwargs["initial_learning_rate"] = self.learning_rate
                    lrskwargs["decay_steps"] = int(self.learning_rate_scheduler_decay_steps)
                    lrskwargs["alpha"] = self.learning_rate_scheduler_cosine_alpha
                elif self.learning_rate_scheduler == "CosineDecayRestarts":
                    lrskwargs["initial_learning_rate"] = self.learning_rate
                    lrskwargs["first_decay_steps"] = int(self.learning_rate_scheduler_decay_steps)
                    lrskwargs["alpha"] = self.learning_rate_scheduler_cosine_alpha
                    lrskwargs["t_mul"] = self.learning_rate_scheduler_cosine_restarts_tmul
                    lrskwargs["m_mul"] = self.learning_rate_scheduler_cosine_restarts_mmul
                else:
                    raise ValueError(f"Untreated learning rate scheduler: {self.learning_rate_scheduler}")

                learning_rate_scheduler = LRSCHEDULERS[self.learning_rate_scheduler](**lrskwargs)
                self.logger.info(f"Learning Rate Scheduler {learning_rate_scheduler.__class__.__name__} was initialized with {lrskwargs}")

            # configure optimizer
            optim_kwargs = {
                "learning_rate": learning_rate_scheduler if self.learning_rate_scheduler != "ReduceLROnPlateau" else self.learning_rate
            }
            if self.optimizer.lower() in ["adam", "adamw", "adamax", "nadam"]:
                optim_kwargs["beta_1"] = 1 - self.anti_momentum_rate
                optim_kwargs["beta_2"] = 1 - self.anti_grad_var_rate
            elif self.optimizer.lower() == "adadelta":
                optim_kwargs["rho"] = 1 - self.anti_grad_var_rate
            elif self.optimizer.lower() == "sgd":
                optim_kwargs["momentum"] = 1 - self.anti_momentum_rate
            elif self.optimizer.lower() == "rmsprop":
                optim_kwargs["momentum"] = 1 - self.anti_momentum_rate
                optim_kwargs["rho"] = 1 - self.anti_grad_var_rate
            optimizer = OPTIMIZERS[self.optimizer](**optim_kwargs)
            self.logger.info(f"Optimizer is {optimizer.__class__.__name__} initialized with {optim_kwargs}")

            iteration_curve_callback = IterationCurveCallback(
                workflow=self,
                timer=self.timer,
                data=dict(
                    # train=dict(X=X, y=np.argmax(y, axis=1)),  # assign the class with the highest true probability (1 except for if data augmentation is used)
                    train=dict(X=X, y=y),
                    val=dict(X=X_valid, y=y_valid),
                    test=dict(X=X_test, y=y_test),
                ),
                logger=self.logger,
                epoch_schedule=self.epoch_schedule,
            )
            if self.num_epochs not in iteration_curve_callback.schedule:
                self.logger.warning(
                    f"Last epoch {self.num_epochs} should be in the schedule covered by the IterationCurveCallback but is not. "
                    f"Schedule is {iteration_curve_callback.schedule}"
                )

            # define callbacks
            callbacks = [
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.EarlyStopping(patience=self.num_epochs_patience),
            ]
            if self.learning_rate_scheduler == "ReduceLROnPlateau":
                self.logger.info(f"Adding {learning_rate_scheduler} as callback")
                callbacks.append(learning_rate_scheduler)

            base_optimizer = optimizer
            if self.lookahead:
                optimizer = Lookahead(
                    optimizer,
                    learning_rate=self.lookahead_learning_rate,
                    la_steps=self.lookahead_num_steps
                )

            # set up weight average callback
            if self.weight_averaging:
                callbacks.append(WA(
                    start_epoch=2,
                    cycle_length=self.weight_averaging_cycle_length,
                    logger=self.logger)
                )

            if self.snapshot_ensemble:
                self.snapshot_callback = Snapshot(
                    workflow=self,
                    optimizer=base_optimizer,
                    reset_weights=self.snapshot_ensemble_reset_weights,
                    period_init=self.snapshot_ensemble_period_init,
                    period_increase=self.snapshot_ensemble_period_increase,
                    logger=self.logger
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
                from lcdb.workflow.keras._augmentation import CutOutAugmentation
                data_augmenters.append(CutOutAugmentation(
                    probability_of_cut=self.data_augmentation_cutout_patch_ratio,
                    random_state=self.random_state
                ))
            elif self.data_augmentation == "mixup":
                from lcdb.workflow.keras._augmentation import MixUpAugmentation
                data_augmenters.append(MixUpAugmentation(random_state=self.random_state))
            elif self.data_augmentation == "cutmix":
                from lcdb.workflow.keras._augmentation import CutMixAugmentation
                data_augmenters.append(CutMixAugmentation(random_state=self.random_state))

            # create log message for the usage of data augmentation 
            if self.data_augmentation is not None:
                self.logger.info(
                    f"Using data augmentation: {self.data_augmentation} with augmenters: {data_augmenters}"
                )
            else:
                self.logger.info("No data augmentation is used.")

            # data generator for augmentation
            train_generator = AugmentDataGenerator(
                                X, y, batch_size=self.batch_size, 
                                augmenters=data_augmenters,
                                encode_label_vector=self._encode_label_vector,
                                shuffle=self.shuffle_each_epoch,
                                random_state=self.random_state
                            )

            # now fit model
            lines = []
            self.learner.summary(print_fn=lambda x: lines.append(x))
            summary_text = "\n".join(lines)
            self.logger.info(f"Compiled model is\n{summary_text}.")
            
            # Check where model variables are placed
            self.logger.info(
                f"Total number of parameters is {self.learner.count_params()}. Those are placed as follows: {''.join(['\n\t' + str(v.name) + " is placed on device " + str(v.handle.device) for v in self.learner.trainable_variables])}"
            )
            tf.debugging.set_log_device_placement(True)

            self.learner.fit(
                train_generator,
                epochs=self.num_epochs,
                shuffle=False, # shuffling is done by the generator to control the random seed
                validation_data=(X_valid[mask_valid], self._encode_label_vector(y_valid[mask_valid])),
                callbacks=callbacks,
                verbose=0,
            )

            # if both weight averaging and batch normalization are active, do a full forward pass over all instances to adjust the batch normalization layers
            if self.weight_averaging and self.batch_norm:
                self.logger.info(f"Doing additional forward pass over all batches to adjust batch norm layers for weight reset performed by WA.")
                num_batches = 0
                for batch_idx in range(len(train_generator)):
                    x_batch, _ = train_generator[batch_idx]
                    self.learner(x_batch, training=True)
                    num_batches += 1
                    self.logger.debug(f"Finished {num_batches}-th forward pass with batch of shape {x_batch.shape} for batch norm layer adjustment performed by WA.")
                self.logger.info(f"Finished {num_batches} forward passes for batch norm layer adjustment performed by WA.")

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
                X, batch_size=min(len(X), self.batch_size), verbose=0
            )
            y_pred_proba.append(y_pred_proba_model)

        # average probabilities
        y_pred_proba = np.array(y_pred_proba)
        if np.any(np.isnan(y_pred_proba)):
            raise RuntimeError(f"There are NAN values in the NN prediction!\n{y_pred_proba}. Input was:\n{X}")
        self.logger.debug(f"Creating prediction based on {len(y_pred_proba)} models with shape {y_pred_proba[0].shape}.")
        if self.snapshot_ensemble and len(self.snapshot_callback.checkpoint_models) > 1 and np.sum(np.var(y_pred_proba, axis=0)) == 0:
            self.logger.warning(f"snapshot ensemble is configured but there is no variance in the outputs. There should be some variance unless all predictions are identical!")
        return y_pred_proba.mean(axis=0)

    def _predict_with_proba_after_transform(self, X, use_snapshot_ensemble=False):

        # obtain probabilistic prediction from snapshot ensemble
        y_pred_proba = self._predict_proba_after_transform(X, use_snapshot_ensemble=use_snapshot_ensemble)

        # derive predictions
        y_pred = self._decode_label_vector(y_pred_proba.argmax(axis=1))

        return y_pred, y_pred_proba

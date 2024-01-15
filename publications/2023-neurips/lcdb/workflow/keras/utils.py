import platform
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def is_arm_mac():
    return platform.system() == "Darwin" and platform.processor() == "arm"


# !On MacOS arm64, some optimizers are not available and some performance issues exist.
if is_arm_mac():
    OPTIMIZERS = {
        "SGD": tf.keras.optimizers.SGD,
        "RMSprop": tf.keras.optimizers.RMSprop,
        "Adam": tf.keras.optimizers.Adam,
        "Adadelta": tf.keras.optimizers.Adadelta,
        "Adagrad": tf.keras.optimizers.Adagrad,
        "Adamax": tf.keras.optimizers.Adamax,
        "Nadam": tf.keras.optimizers.Nadam,
        "Ftrl": tf.keras.optimizers.Ftrl,
    }
else:
    OPTIMIZERS = {
        "SGD": tf.keras.optimizers.SGD,
        "RMSprop": tf.keras.optimizers.RMSprop,
        "Adam": tf.keras.optimizers.Adam,
        "AdamW": tf.keras.optimizers.AdamW,
        "Adadelta": tf.keras.optimizers.Adadelta,
        "Adagrad": tf.keras.optimizers.Adagrad,
        "Adamax": tf.keras.optimizers.Adamax,
        "Adafactor": tf.keras.optimizers.Adafactor,
        "Nadam": tf.keras.optimizers.Nadam,
        "Ftrl": tf.keras.optimizers.Ftrl,
    }

ACTIVATIONS = [
    "none",
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh",
    "selu",
    "elu",
    "exponential",
]

REGULARIZERS = {
    "none": lambda x: None,
    "L1": lambda x: tf.keras.regularizers.L1(l1=x),
    "L2": lambda x: tf.keras.regularizers.L2(l2=x),
    "L1L2": lambda x: tf.keras.regularizers.L1L2(l1=x, l2=x),
}

INITIALIZERS = [
    "random_normal",
    "random_uniform",
    "truncated_normal",
    "zeros",
    "ones",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "orthogonal",
    "variance_scaling",
]


def count_params(model: tf.keras.Model) -> dict:
    """Evaluate the number of parameters of a Keras model.

    Args:
        model (tf.keras.Model): a Keras model.

    Returns:
        dict: a dictionary with the number of trainable ``"num_parameters_train"`` and
        non-trainable parameters ``"num_parameters_not_train"``.
    """
    num_parameters_train = int(
        np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    )
    num_parameters_not_train = int(
        np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    )
    return {
        "num_parameters_not_train": num_parameters_not_train,
        "num_parameters_train": num_parameters_train,
    }

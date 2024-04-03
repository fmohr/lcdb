import platform

import keras
import numpy as np


def is_arm_mac():
    return platform.system() == "Darwin" and platform.processor() == "arm"


# !On MacOS arm64, some optimizers are not available and some performance issues exist.
if is_arm_mac():
    OPTIMIZERS = {
        "SGD": keras.optimizers.SGD,
        "RMSprop": keras.optimizers.RMSprop,
        "Adam": keras.optimizers.Adam,
        "Adadelta": keras.optimizers.Adadelta,
        "Adagrad": keras.optimizers.Adagrad,
        "Adamax": keras.optimizers.Adamax,
        "Nadam": keras.optimizers.Nadam,
        "Ftrl": keras.optimizers.Ftrl,
    }
else:
    OPTIMIZERS = {
        "SGD": keras.optimizers.SGD,
        "RMSprop": keras.optimizers.RMSprop,
        "Adam": keras.optimizers.Adam,
        "AdamW": keras.optimizers.AdamW,
        "Adadelta": keras.optimizers.Adadelta,
        "Adagrad": keras.optimizers.Adagrad,
        "Adamax": keras.optimizers.Adamax,
        "Adafactor": keras.optimizers.Adafactor,
        "Nadam": keras.optimizers.Nadam,
        "Ftrl": keras.optimizers.Ftrl,
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
    "L1": lambda x: keras.regularizers.L1(l1=x),
    "L2": lambda x: keras.regularizers.L2(l2=x),
    "L1L2": lambda x: keras.regularizers.L1L2(l1=x, l2=x),
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


def count_params(model: keras.Model) -> dict:
    """Evaluate the number of parameters of a Keras model.

    Args:
        model (keras.Model): a Keras model.

    Returns:
        dict: a dictionary with the number of trainable ``"num_parameters_train"`` and
        non-trainable parameters ``"num_parameters_not_train"``.
    """
    num_parameters_train = int(
        np.sum([np.prod(v.shape) for v in model.trainable_weights])
    )
    num_parameters_not_train = int(
        np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    )
    return {
        "num_parameters_not_train": num_parameters_not_train,
        "num_parameters_train": num_parameters_train,
    }

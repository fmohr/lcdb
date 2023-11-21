import platform
import tensorflow as tf


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

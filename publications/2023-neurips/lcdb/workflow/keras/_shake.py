from keras import backend as K
from keras.layers import Layer
from tensorflow.random import uniform
from tensorflow import custom_gradient, GradientTape, shape
import tensorflow as tf


class ShakeShake(Layer):
    """ Shake-Shake-Image Layer """

    def __init__(self):
        super().__init__()

    def call(self, x, training=False):

        def shake_shake_combine(x1, x2):
            if training:

                batch_size = shape(x1)[0]

                # Forward mixing coefficient
                alpha = uniform([batch_size, 1,], 0, 1)

                # Backward mixing coefficient
                beta = uniform([batch_size, 1], 0, 1)

                @custom_gradient
                def shake_shake(x1, x2):

                    y = alpha * x1 + (1 - alpha) * x2

                    def grad(dy):
                        grad_x1 = dy * beta
                        grad_x2 = dy * (1 - beta)
                        return grad_x1, grad_x2

                    return y, grad

                return shake_shake(x1, x2)
            else:
                return 0.5 * (x1 + x2)

        return shake_shake_combine(x[0], x[1])

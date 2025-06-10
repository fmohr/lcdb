from keras import backend as K
from keras.layers import Layer
from tensorflow.random import uniform
from tensorflow import custom_gradient, GradientTape, shape
import tensorflow as tf


class ShakeShake(Layer):

    def __init__(self, seed):
        super().__init__()
        self.random_gen = tf.random.Generator.from_seed(seed)

    def call(self, x, training=False):

        def shake_shake_combine(x1, x2):
            if training:

                batch_size = shape(x1)[0]

                # Forward mixing coefficient
                alpha = self.random_gen.uniform([batch_size, 1,], 0, 1)

                # Backward mixing coefficient
                beta = self.random_gen.uniform([batch_size, 1], 0, 1)

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


class ShakeDrop(Layer):
    def __init__(self, seed, p_drop):
        super().__init__()
        self.random_gen = tf.random.Generator.from_seed(seed)
        self.p_drop = p_drop  # Drop probability

    def call(self, inputs, training=False):

        def shake_drop(x):
            if training:
                batch_size = shape(x)[0]

                # Binary gate: 1 = keep branch, 0 = drop
                gate = tf.cast(self.random_gen.uniform([batch_size, 1], 0, 1) > self.p_drop, tf.float32)

                # Forward and backward mixing
                alpha = self.random_gen.uniform([batch_size, 1], 0, 1)
                beta = self.random_gen.uniform([batch_size, 1], 0, 1)

                @custom_gradient
                def shake_drop(residual):
                    
                    # Expand to match shape if needed
                    gate_f = alpha * gate
                    out = gate_f * residual

                    def grad(dy):
                        gate_b = beta * gate
                        grad_residual = dy * gate_b
                        return grad_residual

                    return out, grad

                return shake_drop(x)
            else:
                return (1 - self.p_drop) * x

        return shake_drop(inputs)

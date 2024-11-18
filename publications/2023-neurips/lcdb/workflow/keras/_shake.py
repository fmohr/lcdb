from keras import backend as K
from keras.layers import Layer


class ShakeShake(Layer):
    """ Shake-Shake-Image Layer """

    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x):

        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x
        # create alpha and beta
        batch_size = K.shape(x1)[0]
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        beta = K.random_uniform((batch_size, 1, 1, 1))

        # shake-shake during training phase
        def x_shake():
            return beta * x1 + (1 - beta) * x2 + K.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)
        # even-even during testing phase

        def x_even():
            return 0.5 * x1 + 0.5 * x2
        return K.in_train_phase(x_shake, x_even)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

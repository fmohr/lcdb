import keras.ops as ops
from keras.optimizers import Optimizer
from keras.src.backend import convert_to_numpy
from .utils import serialize_object, deserialize_object
import tensorflow as tf


class Lookahead(Optimizer):
    '''Tensorflow implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''

    def __init__(self,
                 optimizer,
                 learning_rate=0.8,
                 la_steps=5,
                 weight_decay=None,
                 clipnorm=None,
                 global_clipnorm=None,
                 clipvalue=None,
                 use_ema= False,
                 ema_momentum= 0.99,
                 ema_overwrite_frequency=None,
                 loss_scale_factor= None,
                 gradient_accumulation_steps= None,
                 name="Lookahead"
                 ):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        """
        super().__init__(learning_rate=learning_rate, name=name)
        self.optimizer = optimizer
        self._total_la_steps = la_steps
        self.phis = None

    def build(self, variables):
        super().build(variables)
        self.optimizer.build(variables)

        # Create slots for the cached parameters.
        self.phis = []
        for variable in variables:
            phi = self.add_variable_from_reference(
                reference_variable=variable, name="phi"
            )
            self.assign(phi, variable)
            self.phis.append(phi)

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        self.optimizer._backend_update_step(grads, trainable_variables, learning_rate)
        super()._backend_update_step(grads, trainable_variables, learning_rate)

    def update_step(self, gradient, variable, learning_rate):

        # update the actual parameters (inner loop, update of thetas)
        self.optimizer.update_step(gradient, variable, self.optimizer.learning_rate)

        tf.print("VAR after update is")
        tf.print(variable)

        # create condition to check whether this iteration is one in which the outer loop logic should be executed
        local_step = self.iterations + 1
        sync_cond = ops.equal(
            ops.floor_divide(local_step, self._total_la_steps) * self._total_la_steps,
            local_step
        )

        # define the step-back logic and apply it conditionally
        phi = self.phis[self._get_variable_index(variable)]
        step_back = phi + learning_rate * (variable - phi)  # the learning rate here is the one of Lookahead

        # update phi values (cache values) if this iteration include the outer loop
        self.assign(
            phi,
            ops.where(sync_cond, step_back, phi)
        )

        # in that case, also over-write the actual parameter values by that same values
        self.assign(
            variable,
            ops.where(sync_cond, step_back, variable)
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "optimizer": serialize_object(self.optimizer),
            "learning_rate": convert_to_numpy(self.learning_rate),
            "la_steps": self._total_la_steps
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["optimizer"] = deserialize_object(config["optimizer"])
        return cls(**config)

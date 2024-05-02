from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
import keras.ops as ops
from keras.optimizers import Optimizer


class Lookahead(Optimizer):
    '''Tensorflow implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, name="Lookahead"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        """
        super().__init__(learning_rate=la_alpha, name=name)
        self.optimizer = optimizer
        self._total_la_steps = la_steps
        self.phis = None

    def build(self, variables):
        super().build(variables)
        self.optimizer.build(variables)

        # Create slots for the cached parameters.
        self.phis = []
        for variable in variables:
            self.phis.append(
                self.add_variable_from_reference(
                    reference_variable=variable, name="phi"
                )
            )

    def update_step(self, gradient, variable, learning_rate):

        # update the actual parameters (inner loop, update of thetas)
        self.optimizer.update_step(gradient, variable, self.optimizer.learning_rate)

        # create condition to check whether this iteration is one in which the outer loop logic should be executed
        local_step = ops.cast(self.iterations + 1, tf.dtypes.int64)
        sync_cond = ops.equal(
            ops.floor_divide(local_step, self._total_la_steps) * self._total_la_steps,
            local_step
        )

        # define the step-back logic and apply it conditionally
        phi = self.phis[self._get_variable_index(variable)]
        step_back = phi + learning_rate * (variable - phi)  # the learning rate here is the one of Lookahead
        with tf.control_dependencies([step_back]):

            # update phi values (cache values) if this iteration include the outer loop
            self.assign(
                phi,
                tf.where(sync_cond, step_back, phi)
            )

            # in that case, also over-write the actual parameter values by that same values
            self.assign(
                variable,
                tf.where(sync_cond, step_back, variable)
            )

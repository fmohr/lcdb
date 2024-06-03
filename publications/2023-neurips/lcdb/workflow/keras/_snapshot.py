import math
import os

from keras.callbacks import Callback
from keras.src.backend import convert_to_numpy
from keras.models import clone_model

import numpy as np


class Snapshot(Callback):

    def __init__(
            self,
            workflow,
            optimizer,
            period=20,
            reset_weights=False,
            verbose=0
    ):

        super(Snapshot, self).__init__()
        self.workflow = workflow
        self.optimizer = optimizer
        self.verbose = verbose
        self.period = period
        self.reset_weights = reset_weights

        self.checkpoint_models = []

    def _set_lr(self, lr):
        self.model.optimizer.learning_rate = lr

    def _get_lr(self):
        return float(convert_to_numpy(self.model.optimizer.learning_rate))

    def _reset_weights(self):

        # Reset weights randomly using their own initializers
        for layer in self.model.layers:
            seed = np.random.randint(0, 10000)
            if hasattr(layer, 'kernel_initializer'):
                if hasattr(layer.kernel_initializer, "seed"):
                    layer.kernel_initializer.seed = seed
                new_kernel = layer.kernel_initializer(shape=layer.kernel.shape)
                if hasattr(layer, 'bias_initializer') and layer.use_bias:
                    if hasattr(layer.bias_initializer, "seed"):
                        layer.bias_initializer.seed = seed
                    new_bias = layer.bias_initializer(shape=layer.bias.shape)
                    layer.set_weights([new_kernel, new_bias])
                else:
                    layer.set_weights([new_kernel])

    def on_epoch_end(self, epoch, logs=None):

        # only do something at the end of each cycle
        if epoch == 0 or (epoch + 1) % self.period != 0:
            return

        # save model for this cycle
        model_copy = clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        self.checkpoint_models.append(model_copy)

        # Resetting the weights if configured (the LR is implicitly reset at the beginning of the next epoch)
        if self.reset_weights:
            self._reset_weights()

    def on_epoch_begin(self, epoch, logs=None):

        # adjust learning rate through cyclic cosine annealing
        if epoch > 0:
            lr = math.pi * (epoch % self.period) / self.period
            lr = self.base_lr / 2 * (math.cos(lr) + 1)
            self._set_lr(lr)
            print(f"Learning rate is now {convert_to_numpy(self.model.optimizer.learning_rate)}")

    def on_test_begin(self, logs=None):
        self.workflow.use_snapshot_models_for_prediction = True

    def on_test_end(self, logs=None):
        self.workflow.use_snapshot_models_for_prediction = False

    def on_predict_begin(self, logs=None):
        self.workflow.use_snapshot_models_for_prediction = True

    def on_predict_end(self, logs=None):
        self.workflow.use_snapshot_models_for_prediction = False

    def set_model(self, model):
        super().set_model(model)

        # Get initial learning rate
        self.base_lr = float(self._get_lr())

        self._reset_weights()

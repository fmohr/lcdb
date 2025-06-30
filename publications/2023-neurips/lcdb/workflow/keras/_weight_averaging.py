from keras.callbacks import Callback
import keras.backend as K


class WA(Callback):

    def __init__(self, start_epoch, cycle_length, logger):

        super().__init__()
        self.start_epoch = start_epoch - 1 # internally these start at 0
        self.cycle_length = cycle_length
        self.logger = logger

        if start_epoch < 2:
            raise ValueError('"swa_start" attribute cannot be lower than 2.')

    def on_train_begin(self, logs=None):
        self.epochs = self.params.get("epochs")

        if self.start_epoch >= self.epochs - 1:
            raise ValueError('"swa_start" attribute must be lower than "epochs".')

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

        # if SWA starts now, retrieve current weights in a deep copy
        if epoch == self.start_epoch:
            self.wa_weights = self.model.get_weights()
            self.logger.info(f"Starting weight averaging in epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):

        # average the weights if we are in an SWA epoch
        epoch_after_start = epoch - self.start_epoch
        if epoch_after_start % self.cycle_length == 0:# and not self.is_batch_norm_epoch:
            n_models = epoch_after_start / self.cycle_length
            self.wa_weights = [
                (w_wa * n_models + w) / (n_models + 1)
                for w_wa, w in zip(self.wa_weights, self.model.get_weights())
            ]
            num_updated_weights = sum([w.size for w in self.wa_weights])
            self.logger.info(f"Updated {num_updated_weights} average weights in WA callback.")

    def on_train_end(self, logs=None):

        # replace weights w with average weights w_swa. Updating batch norm must happen outside in an explicit forward pass.
        num_updated_weights = sum([w.size for w in self.wa_weights])
        self.model.set_weights(self.wa_weights)
        self.logger.info(f"{num_updated_weights} model weights overwritten with WA weights.")

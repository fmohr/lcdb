import tensorflow as tf
from tensorflow import data as tf_data
from keras.random import gamma as tf_random_gamma
import keras.ops as ops
import numpy as np


class Augmenter:

    def __init__(self, replace=False):
        self.replace = replace

    def augment(self, X, y):
        raise NotImplementedError


class RandomnessBasedAugmenter(Augmenter):

    def __init__(self, random_state, **kwargs):
        super().__init__(**kwargs)
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState() if random_state is None else np.random.RandomState(random_state)


class CutOutAugmentation(RandomnessBasedAugmenter):

    def __init__(self, probability_of_cut, **kwargs):
        super().__init__(**kwargs)
        self.probability_of_cut = probability_of_cut

    def augment(self, X, y):
        indices = list(range(X.shape[0]))
        n_instances = len(indices)
        n_cells = n_instances * X.shape[1]
        random_binary_mask = self.random_state.random(size=n_cells).reshape((n_instances, -1)) <= self.probability_of_cut

        # cut mix X
        X[random_binary_mask] = 0
        return X, y


class MixUpAugmentation(RandomnessBasedAugmenter):

    def __init__(self, split_point_distribution=None, **kwargs):
        super().__init__(**kwargs)

        if split_point_distribution is not None:
            self.split_point_distribution = split_point_distribution
        else:
            self.split_point_distribution = self.sample_beta_distribution

    def sample_beta_distribution(self, size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf_random_gamma(shape=[size], alpha=concentration_1, seed=self.random_state.randint(10**10))
        gamma_2_sample = tf_random_gamma(shape=[size], alpha=concentration_0, seed=self.random_state.randint(10**10))
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def augment(self, X, y):
        indices = list(range(X.shape[0]))
        random_partner_indices = self.random_state.choice(indices, size=len(indices), replace=False)
        cross_over = tf.convert_to_tensor(self.split_point_distribution(size=len(indices))).reshape(-1, 1)

        # mix up X
        X = tf.convert_to_tensor(X)
        X_right = X[random_partner_indices]
        X_after = X * cross_over + X_right * (1 - cross_over)

        # mix up y
        y = tf.convert_to_tensor(y)
        y_right = y[random_partner_indices]
        y_after = y * cross_over + y_right * (1 - cross_over)

        return X_after, y_after


class CutMixAugmentation(RandomnessBasedAugmenter):

    def __init__(self, split_point_distribution=None, **kwargs):
        super().__init__(**kwargs)

        if split_point_distribution is not None:
            self.split_point_distribution = split_point_distribution
        else:
            self.split_point_distribution = self.sample_beta_distribution

    def sample_beta_distribution(self, size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf_random_gamma(shape=[size], alpha=concentration_1, seed=self.random_state.randint(10**10))
        gamma_2_sample = tf_random_gamma(shape=[size], alpha=concentration_0, seed=self.random_state.randint(10**10))
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def augment(self, X, y):
        indices = list(range(X.shape[0]))
        n_instances = len(indices)
        n_cells = n_instances * X.shape[1]
        random_partner_indices = self.random_state.choice(indices, size=n_instances, replace=False)
        random_binary_mask = self.random_state.randint(low=0, high=2, size=n_cells).reshape((n_instances, -1))
        lambdas = tf.convert_to_tensor(self.split_point_distribution(size=n_instances)).reshape(-1, 1)

        # cut mix X
        X = tf.convert_to_tensor(X)
        X_right = X[random_partner_indices]
        X_after = X * random_binary_mask + X_right * (1 - random_binary_mask)

        # mix up y
        y = tf.convert_to_tensor(y)
        y_right = y[random_partner_indices]
        y_after = y * lambdas + y_right * (1 - lambdas)

        return X_after, y_after

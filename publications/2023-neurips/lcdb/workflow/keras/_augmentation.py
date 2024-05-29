from tensorflow import data as tf_data
from keras.random import gamma as tf_random_gamma
import keras.ops as ops


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf_random_gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf_random_gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    X_1, y_1 = ds_one
    X_2, y_2 = ds_two
    batch_size = X_1.shape[0]

    print(X_1.shape)

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    #x_l = ops.reshape(l, X_1.shape)
    #y_l = ops.reshape(l, y_1.shape)
    print(l)
    #print(x_l)
    #print(y_l)

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    X = X_1 * x_l + X_2 * (1 - x_l)
    y = y_1 * y_l + y_2 * (1 - y_l)
    return X, y


def augment_with_mixup(X, y):

    print(X.shape)

    # get two randomly shuffled version of the datasets and zip them
    train_ds_one = (
        tf_data.Dataset.from_tensor_slices((X, y))
        .shuffle(X.shape[0])
    )
    train_ds_two = (
        tf_data.Dataset.from_tensor_slices((X, y))
        .shuffle(X.shape[0])
    )
    train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))

    # now mix-up pair instances
    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2)
        #num_parallel_calls=AUTO,
    )
    return train_ds_mu
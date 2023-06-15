import logging
import numpy as np
import openml
import sklearn



def get_openml_dataset(openmlid):
    """Returns (X, y) arrays from the corresponding OpenML dataset ID."""
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]

    # prepare label column as numpy array
    logging.info(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
    y = np.array(df[ds.default_target_attribute].values)
    if y.dtype != int:
        y_int = np.zeros(len(y)).astype(int)
        vals = np.unique(y)
        for i, val in enumerate(vals):
            mask = y == val
            y_int[mask] = i
        y = y_int

    logging.info(f"Data is of shape {X.shape}.")
    return X, y




def compile_learner(learner, X, y, drop_first=True, binarize_sparse=False):
    pre = get_mandatory_preprocessing(
        X, y, binarize_sparse=binarize_sparse, drop_first=drop_first
    )
    if not pre:
        return learner
    return sklearn.pipeline.Pipeline(pre + [("classifier", learner)])


def get_schedule(dataset_size):
    min_exp = 4
    max_train_size = int(max(int(dataset_size * 0.9) * 0.9, dataset_size - 10000))
    max_exp = np.log(max_train_size) / np.log(2)
    max_exp_int = (int(np.floor(max_exp)) - min_exp) * 2
    anchors = [
        int(np.round(2 ** (min_exp + 0.5 * exp))) for exp in range(0, max_exp_int + 1)
    ]
    if anchors[-1] != max_train_size:
        anchors.append(max_train_size)
    return anchors
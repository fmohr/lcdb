import logging
import numpy as np
import openml
import sklearn
from functools import lru_cache


@lru_cache(maxsize=2)
def get_openml_dataset_and_check(openmlid):
    # load data
    binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
    drop_first = False  # openmlid not in [3] # drop first cannot be used in datasets with some very rare categorical values
    print(f"Reading dataset. Will be binarized sparsely: {binarize_sparse}")
    X, y = get_openml_dataset(openmlid)
    y = np.array([str(e) for e in y])  # make sure that labels are strings
    print(
        f"ready. Dataset shape is {X.shape}, label column shape is {y.shape}. Now running the algorithm"
    )
    if X.shape[0] <= 0:
        raise Exception("Dataset size invalid!")
    if X.shape[0] != len(y):
        raise Exception("X and y do not have the same size.")

    return X, y, binarize_sparse, drop_first

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
import numpy as np
import pandas as pd
import openml
import os
import logging

import time

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sklearn
from sklearn import *

eval_logger = logging.getLogger("evalutils")


def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    num_rows = len(df)
        
    # prepare label column as numpy array
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
    y = np.array(df[ds.default_target_attribute].values)
    if y.dtype != int:
        y_int = np.zeros(len(y)).astype(int)
        vals = np.unique(y)
        for i, val in enumerate(vals):
            mask = y == val
            y_int[mask] = i
        y = y_int
        
    print(f"Data is of shape {X.shape}.")
    return X, y

def get_outer_split(X, y, seed):
    test_samples_at_90_percent_training = int(X.shape[0] * 0.1)
    return sklearn.model_selection.train_test_split(X, y, train_size = 0.9, random_state=seed, stratify=y)

def get_inner_split(X, y, outer_seed, inner_seed):
    X_learn, X_test, y_learn, y_test = get_outer_split(X, y, outer_seed)
    validation_samples_at_90_percent_training = int(X_learn.shape[0] * 0.1)
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X_learn, y_learn, train_size = 0.9, random_state=inner_seed, stratify=y_learn)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed, monotonic):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed, inner_seed)
    if not monotonic: # shuffle index set if the train fold should not be monotonic.
        rs = np.random.RandomState(inner_seed)
        indices = rs.choice(range(X_train.shape[0]), X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # check that anchor is not bigger than allowed
    if anchor > X_train.shape[0]:
        raise ValueError(f"Invalid anchor {anchor} when available training instances are only {X_train.shape[0]}.")
    return X_train[:anchor], X_valid, X_test, y_train[:anchor], y_valid, y_test

def get_mandatory_preprocessing(X, y, binarize_sparse = False, drop_first = True):
    
    # determine fixed pre-processing steps for imputation and binarization
    types = [set([type(v) for v in r]) for r in X.T]
    numeric_features = [c for c, t in enumerate(types) if len(t) == 1 and list(t)[0] != str]
    numeric_transformer = Pipeline([("imputer", sklearn.impute.SimpleImputer(strategy="median"))])
    categorical_features = [i for i in range(X.shape[1]) if i not in numeric_features]
    missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
    eval_logger.info(f"There are {len(categorical_features)} categorical features, which will be binarized.")
    eval_logger.info(f"Missing values for the different attributes are {missing_values_per_feature}.")
    if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
        handle_unknown = "error" if drop_first else "ignore"
        categorical_transformer = Pipeline([
            ("imputer", sklearn.impute.SimpleImputer(strategy="most_frequent")),
            ("binarizer", sklearn.preprocessing.OneHotEncoder(drop='first' if drop_first else None, handle_unknown=handle_unknown, sparse=binarize_sparse)),
        ])
        return [("impute_and_binarize", ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        ))]
    else:
        return []
    
def compile_learner(learner, X, y, drop_first = True, binarize_sparse = False):
    pre = get_mandatory_preprocessing(X, y, binarize_sparse = binarize_sparse, drop_first = drop_first)
    if not pre:
        return learner
    return sklearn.pipeline.Pipeline(pre + [("classifier", learner)])

def get_schedule(dataset_size):
    min_exp = 4
    max_train_size = int(max(int(dataset_size * 0.9) * 0.9, dataset_size - 10000))
    max_exp = np.log(max_train_size) / np.log(2)
    max_exp_int = (int(np.floor(max_exp)) - min_exp) * 2
    anchors = [int(np.round(2**(min_exp + 0.5 * exp))) for exp in range(0, max_exp_int + 1)]
    if anchors[-1] != max_train_size:
        anchors.append(max_train_size)
    return anchors


def compute_curve_for_svc(openmlid: int, outer_seed: int, inner_seed: int, train_size: int, monotonic: bool, complexity_constant: float, gamma: float, logger = None):
    
    learner = sklearn.svm.SVC(kernel = "rbf", C = complexity_constant, gamma = gamma)
    return compute_result(openmlid, outer_seed, inner_seed, train_size, monotonic, learner, logger)
    

def compute_result(openmlid: int, outer_seed: int, inner_seed: int, train_size: int, monotonic: bool, learner, logger = None):
    """Compute the learning curve for a given dataset and workflow.

    Args:
        openmlid (int): An OpenML dataset ID.
        outer_seed (int): The seed for the outer split.
        inner_seed (int): The seed for the inner split.
        train_size (int): The size of the training set.
        monotonic (bool): A constraint to enforce the learning curve to be monotonic. # TODO: what does this mean?
        learner (_type_): A instance of a scikit-learn estimtaor.
        logger (_type_, optional): _description_. Defaults to None.

    Raises:
        Exception: _description_
        Exception: _description_
    """
    
    if logger is None:
        logger = logging.getLogger("lcdb.exp")
    
    # CPU
    logger.info("CPU Settings:")
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        logger.info(f"\t{v}: {os.environ.get(v, 'n/a')}")
    
    # load data
    binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
    logger.info(f"Reading dataset. Will be binarized sparsely: {binarize_sparse}")
    X, y = get_dataset(openmlid)
    logger.info(f"ready. Dataset shape is {X.shape}, label column shape is {y.shape}. Now running the algorithm")
    if X.shape[0] <= 0:
        raise Exception("Dataset size invalid!")
    if X.shape[0] != len(y):
        raise Exception("X and y do not have the same size.")
    
    # get workflow
    compiled_learner_base = compile_learner(learner, X, y, drop_first = False)
    
    labels = sorted(np.unique(y))
    
    # get data
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(X, y, train_size, outer_seed, inner_seed, monotonic)
    logger.info(f"Data created for anchor {train_size}. Sizes are {X_train.shape} (train), {X_valid.shape} (valid), and {X_test.shape} (test).")

    # run selector
    compiled_learner = sklearn.base.clone(compiled_learner_base)
    time_start = time.time()
    compiled_learner.fit(X_train, y_train)
    traintime = time.time() - time_start

    #labels_tmp = compiled_learner.classes_
    #if labels is None:
#            labels = labels_tmp
#        else:
#            if np.any(labels != labels_tmp):
#                raise ValueError(f"Labels don't match. Originals are {labels}. Learned by workflow are {labels_tmp}")

    logger.info(f"SVM fitted after {np.round(traintime, 2)}s. Now obtaining predictions.")

    # compute confusion matrices
    start = time.time()
    y_hat_train = compiled_learner.predict(X_train)
    predict_time_train = time.time() - start
    start = time.time()
    y_hat_valid = compiled_learner.predict(X_valid)
    predict_time_valid = time.time() - start
    start = time.time()
    y_hat_test = compiled_learner.predict(X_test)
    predict_time_test = time.time() - start

    cm_train = sklearn.metrics.confusion_matrix(y_train, y_hat_train, labels = labels)
    cm_valid = sklearn.metrics.confusion_matrix(y_valid, y_hat_valid, labels = labels)
    cm_test = sklearn.metrics.confusion_matrix(y_test, y_hat_test, labels = labels)
        
    return labels, cm_train, cm_valid, cm_test, traintime, predict_time_train, predict_time_valid, predict_time_test
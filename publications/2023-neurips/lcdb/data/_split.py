from typing import Union, Tuple

import numpy as np
import sklearn.model_selection

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import sklearn.impute

def get_outer_split(X, y, seed, ratio=0.1):
    """Returns (X_train, X_test, y_train, y_test) arrays with len(X_test) / len(X) = ratio."""
    return sklearn.model_selection.train_test_split(
        X, y, train_size=1 - ratio, random_state=seed, stratify=y
    )


def get_inner_split(X, y, outer_seed, inner_seed, outer_ratio=0.1, inner_ratio=0.1):
    """Returns (X_train, X_valid, X_test, y_train, y_valid, y_test) arrays with len(X_test) / len(X) = outer_ratio and len(X_valid) / (len(X) - len(X_test)) = inner_ratio."""

    num_instances = X.shape[0]
    max_training_set_size = int((1 - outer_ratio) * (1 - inner_ratio) * num_instances)

    X_learn, X_test, y_learn, y_test = get_outer_split(
        X, y, outer_seed, ratio=outer_ratio
    )
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        X_learn,
        y_learn,
        train_size=max_training_set_size,
        random_state=inner_seed,
        stratify=y_learn,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed, monotonic, valid_prop=0.1, test_prop=0.1):
    """Returns (X_train, X_valid, X_test, y_train, y_valid, y_test) arrays with len(X_test) / len(X) = outer_ratio and len(X_valid) / (len(X) - len(X_test)) = inner_ratio and X_train is truncated at index anchor."""

    X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(
        X, y, outer_seed, inner_seed, inner_ratio=valid_prop, outer_ratio=test_prop
    )
    if not monotonic:  # shuffle index set if the train fold should not be monotonic.
        rs = np.random.RandomState(inner_seed)
        indices = rs.choice(range(X_train.shape[0]), X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

    # check that anchor is not bigger than allowed
    if anchor > X_train.shape[0]:
        raise ValueError(
            f"Invalid anchor {anchor} when available training instances are only {X_train.shape[0]}."
        )
    return X_train[:anchor], X_valid, X_test, y_train[:anchor], y_valid, y_test


def get_mandatory_preprocessing(X, y, binarize_sparse=False, drop_first=True, scaler='minmax'):
    # determine fixed pre-processing steps for imputation and binarization
    types = [set([type(v) for v in r]) for r in X.T]
    numeric_features = [
        c for c, t in enumerate(types) if len(t) == 1 and list(t)[0] != str
    ]

    myscaler = None
    if scaler == 'minmax':
        myscaler = sklearn.preprocessing.MinMaxScaler()
    if scaler == 'standardize':
        myscaler = sklearn.preprocessing.StandardScaler()
    if scaler == 'none':
        numeric_transformer = Pipeline(
            [("imputer", sklearn.impute.SimpleImputer(strategy="median"))]
        )
    else:
        if myscaler == None:
            raise Exception("The scaler %s is not implemented." % scaler)
        numeric_transformer = Pipeline(
            [("imputer", sklearn.impute.SimpleImputer(strategy="median")),
             ("standardscaler", myscaler)]
        )

    categorical_features = [i for i in range(X.shape[1]) if i not in numeric_features]
    missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
    logging.info(
        f"There are {len(categorical_features)} categorical features, which will be binarized."
    )
    logging.info(
        f"Missing values for the different attributes are {missing_values_per_feature}."
    )
    # if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
    if True: # always do preprocessing, because training set may not contain nans by coincidence
        handle_unknown = "error" if drop_first else "ignore"
        categorical_transformer = Pipeline(
            [
                ("imputer", sklearn.impute.SimpleImputer(strategy="most_frequent")),
                (
                    "binarizer",
                    sklearn.preprocessing.OneHotEncoder(
                        drop="first" if drop_first else None,
                        handle_unknown=handle_unknown,
                        sparse=binarize_sparse,
                    ),
                ),
            ]
        )
        return [
            (
                "impute_and_binarize",
                ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numeric_features),
                        ("cat", categorical_transformer, categorical_features),
                    ]
                ),
            )
        ]
    else:
        return []

def random_split_from_array(
    *arrays: Tuple[np.ndarray],
    train_size: float = 0.9,
    stratify_with_targets: bool = False,
    shuffle: bool = True,
    random_state: Union[int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs a random split of the input arrays.

    Args:
        arrays (Tuple[np.ndarray]): arrays to be split.
        train_size (float, optional): proportion of the training out of the whole data, between [0,1]. Defaults to 0.9.
        stratify_with_targets (bool, optional): stratify the split according to label classes. Defaults to False.
        shuffle (bool, optional): randomly shuffle the array before splitting. Defaults to True.
        random_state (Union[int, np.random.RandomState], optional): random state of the shuffling process. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (X_train,  X_test, y_train, y_test) the splitted arrays.
    """
    return sklearn.model_selection.train_test_split(
        *arrays,
        train_size=train_size,
        random_state=random_state,
        stratify=arrays[1] if stratify_with_targets else None,
        shuffle=shuffle,
    )

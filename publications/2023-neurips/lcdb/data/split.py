from typing import Union, Tuple

import numpy as np
import sklearn.model_selection

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import sklearn.impute


def train_valid_test_split(
    X,
    y,
    test_seed,
    valid_seed,
    test_prop=0.1,
    valid_prop=0.1,
    stratify=True,
):
    X_learn, X_test, y_learn, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=test_prop,
        random_state=test_seed,
        stratify=y if stratify else None,
        shuffle=True,
    )
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        X_learn,
        y_learn,
        train_size=int(X.shape[0] * (1 - test_prop - valid_prop)),
        random_state=valid_seed,
        stratify=y_learn if stratify else None,
        shuffle=True,  # TODO: check
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_mandatory_preprocessing(
    X, y, binarize_sparse=False, drop_first=True, scaler="minmax"
):
    # determine fixed pre-processing steps for imputation and binarization
    types = [set([type(v) for v in r]) for r in X.T]
    numeric_features = [
        c for c, t in enumerate(types) if len(t) == 1 and list(t)[0] != str
    ]

    myscaler = None
    if scaler == "minmax":
        myscaler = sklearn.preprocessing.MinMaxScaler()
    if scaler == "standardize":
        myscaler = sklearn.preprocessing.StandardScaler()
    if scaler == "none":
        numeric_transformer = Pipeline(
            [("imputer", sklearn.impute.SimpleImputer(strategy="median"))]
        )
    else:
        if myscaler == None:
            raise Exception("The scaler %s is not implemented." % scaler)
        numeric_transformer = Pipeline(
            [
                ("imputer", sklearn.impute.SimpleImputer(strategy="median")),
                ("standardscaler", myscaler),
            ]
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
    if (
        True
    ):  # always do preprocessing, because training set may not contain nans by coincidence
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

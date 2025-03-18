import warnings

import os
from abc import ABC

import numpy as np
import pandas as pd
from sys import getsizeof
from ConfigSpace import (
    Constant,
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
    EqualsCondition,
    InCondition,
)
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler, LabelEncoder,
)

from ._base_workflow import BaseWorkflow
from lcdb.builder.utils import estimate_memory_consumption_for_dataset

KEY_CAT_ENCODER = "cat_encoder"
KEY_SCALER = "scaler"
KEY_FEATUREGEN = "featuregen"
KEY_FEATUREMAPPER = "decomposition"
KEY_FEATURESELECTOR = "featureselector"

CONFIG_SPACE = ConfigurationSpace(
    name="standard_preprocessing",
    space={
        KEY_CAT_ENCODER: Categorical(
            KEY_CAT_ENCODER, ["onehot", "ordinal"], default="onehot"
        ),
        KEY_SCALER: Categorical(KEY_SCALER, ["none", "minmax", "std"], default="none"),
        KEY_FEATUREGEN: Categorical(KEY_FEATUREGEN, ["none", "poly"], default="none"),
        KEY_FEATUREMAPPER: Categorical(
            KEY_FEATUREMAPPER,
            [
                "none",
                "kernel_pca",  # we just keep this one as it includes "linear pca"
                "lda",
                "fastica",
                "ka_rbf",
                "ka_nystroem",
                "agglomerator",
            ],
            default="none",
        ),
        KEY_FEATURESELECTOR: Categorical(
            KEY_FEATURESELECTOR,
            ["none", "selectp"],
            default="none",
        ),
        # parameters of possible elements of the preprocessing pipeline
        "kernel_pca_kernel": Categorical(
            "kernel_pca_kernel", items=["linear", "rbf"], default="linear"
        ),
        "kernel_pca_n_components": Float(
            "kernel_pca_n_components", bounds=(0.25, 1.0), default=1.0
        ),
        "selectp_percentile": Integer(
            "selectp_percentile", bounds=(25, 100), default=100
        ),
        "poly_degree": Constant("poly_degree", 2),#Integer("poly_degree", bounds=(2, 2), default=2),
        "std_with_std": Categorical("std_with_std", [True, False], default=True),
    },
)

CONFIG_SPACE.add(
    [
        EqualsCondition(
            CONFIG_SPACE["kernel_pca_kernel"],
            CONFIG_SPACE[KEY_FEATUREMAPPER],
            "kernel_pca",
        ),
        EqualsCondition(
            CONFIG_SPACE["kernel_pca_n_components"],
            CONFIG_SPACE[KEY_FEATUREMAPPER],
            "kernel_pca",
        ),
        EqualsCondition(
            CONFIG_SPACE["selectp_percentile"],
            CONFIG_SPACE[KEY_FEATURESELECTOR],
            "selectp",
        ),
        EqualsCondition(
            CONFIG_SPACE["poly_degree"], CONFIG_SPACE[KEY_FEATUREGEN], "poly"
        ),
        EqualsCondition(CONFIG_SPACE["std_with_std"], CONFIG_SPACE[KEY_SCALER], "std"),
    ]
)


class PreprocessedWorkflow(BaseWorkflow, ABC):
    _config_space = CONFIG_SPACE

    def __init__(
        self,
        timer=None,
        logger=None,
        random_state=None,
        memory_limit_in_bytes=None,
        kernel_pca_kernel="linear",
        kernel_pca_n_components=1.0,
        selectp_percentile=100,
        poly_degree=2,
        std_with_std=True,
        raise_exception_on_unsuitable_preprocessor=True,
        **kwargs,
    ):
        super().__init__(timer=timer, logger=logger, random_state=random_state, memory_limit_in_bytes=memory_limit_in_bytes)

        # extract preprocessing hyperparameters
        self.pp_kws = kwargs
        self.pp_pipeline = None

        self.kernel_pca_kernel = kernel_pca_kernel
        self.kernel_pca_n_components = kernel_pca_n_components
        self.selectp_percentile = selectp_percentile
        self.poly_degree = poly_degree
        self.std_with_std = std_with_std
        self.raise_exception_on_unsuitable_preprocessor = raise_exception_on_unsuitable_preprocessor

    @classmethod
    def config_space(
        cls,
        techniques=None,
    ):
        if techniques is None:
            return cls._config_space

        # TODO: update
        cs = ConfigurationSpace()
        hp_names = set(hp.name for hp in cls._config_space.get_hyperparameters())
        unknown_techniques = set(techniques).difference(hp_names)
        if unknown_techniques:
            raise ValueError(
                f"Unknown preprocessing technique keys: {unknown_techniques}"
            )
        for hp in cls._config_space.get_hyperparameters():
            if hp.name in techniques:
                cs.add_hyperparameter(hp)
        return cs
    
    def _anticipate_required_memory_for_fit(self, input_shape, pre_processor):
        if isinstance(pre_processor, PolynomialFeatures):
            poly = pre_processor
            num_created_features = poly._num_combinations(
                n_features=input_shape[1],
                min_degree=0,
                max_degree=poly.degree,
                interaction_only=poly.interaction_only,
                include_bias=poly.include_bias,
            )
            return estimate_memory_consumption_for_dataset((input_shape[0], input_shape[1] + num_created_features))
        
        if isinstance(pre_processor, FeatureAgglomeration) or isinstance(pre_processor, PCA):
            
            # Will test `num_features**2` possible "links" (translating the memory complexity)
            return (
                estimate_memory_consumption_for_dataset(input_shape) +
                estimate_memory_consumption_for_dataset((input_shape[1], input_shape[1])) # n_features²
            )
        
        if isinstance(pre_processor, KernelPCA):
            return estimate_memory_consumption_for_dataset((input_shape[0], input_shape[0]))  # n_samples²
        
        # if no transformation is known, anticipate that the shape will not be changed
        return estimate_memory_consumption_for_dataset(input_shape)
    
    def _anticipate_required_memory_for_transform(self, input_shape, pre_processor):
        if isinstance(pre_processor, PolynomialFeatures):
            poly = pre_processor
            num_created_features = poly._num_combinations(
                n_features=input_shape[1],
                min_degree=0,
                max_degree=poly.degree,
                interaction_only=poly.interaction_only,
                include_bias=poly.include_bias,
            )
            return estimate_memory_consumption_for_dataset((input_shape[0], input_shape[1] + num_created_features))
        
        if isinstance(pre_processor, FeatureAgglomeration) or isinstance(pre_processor, PCA):
            
            # Will test `num_features**2` possible "links" (translating the memory complexity)
            return (
                estimate_memory_consumption_for_dataset(input_shape) +
                estimate_memory_consumption_for_dataset((input_shape[1], input_shape[1])) # n_features²
            )
        
        if isinstance(pre_processor, KernelPCA):
            return estimate_memory_consumption_for_dataset((pre_processor.eigenvectors_.shape[0], input_shape[0]))  # n_train * n_new
        
        # if no transformation is known, anticipate that the shape will not be changed
        return estimate_memory_consumption_for_dataset(input_shape)

    def _transform(self, X, y, metadata):

        self.logger.info(f"Starting data transformation. Configured memory limit: {self.memory_limit_in_bytes // 1024**2}MB")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            if not self.transform_fitted:

                self.logger.debug(f"Fitting the pre-processors of the pipeline.")

                # get further pre-processing steps
                pp_steps = self.get_pp_steps(X, y, metadata, **self.pp_kws)
                for step_name, step_fun in pp_steps:

                    # anticipate memory usage and possibly avoid execution
                    used_memory_bytes = self._anticipate_required_memory_for_fit(X.shape, step_fun)
                    self.logger.info(f"Expected memory usage run pre-processor {step_fun.__class__.__name__}: {used_memory_bytes / (1024**3)}GB.")
                    if used_memory_bytes > self.memory_limit_in_bytes:
                        raise RuntimeError(
                            f"{step_name} ({step_fun.__class__.__name__}) is predicted to consume approximately {used_memory_bytes/(1024**3):.3f} GB. "
                            f"The permitted maximum is {self.memory_limit_in_bytes/(1024**3):.3f} GB!"
                            )

                    # transform the data
                    with self.timer.time(step_name) as node:
                        X = step_fun.fit_transform(X, y=y)
                        node["new_shape"] = {"rows": X.shape[0], "cols": X.shape[1]}
                        self.logger.debug(f"New data shape is {X.shape}")
                self.pp_pipeline = Pipeline(steps=pp_steps)
            else:
                self.logger.debug(f"Starting data transformation with input size {X.shape}")
                for step_name, step_fun in self.pp_pipeline.steps:

                    # anticipate memory usage and possibly avoid execution
                    used_memory_bytes = self._anticipate_required_memory_for_transform(X.shape, step_fun)
                    self.logger.info(f"Expected memory usage run pre-processor {step_fun.__class__.__name__}: {used_memory_bytes / (1024**3)}GB.")
                    if used_memory_bytes > self.memory_limit_in_bytes:
                        raise RuntimeError(
                            f"{step_name} ({step_fun.__class__.__name__}) is predicted to consume approximately {used_memory_bytes/(1024**3):.3f} GB. "
                            f"The permitted maximum is {self.memory_limit_in_bytes/(1024**3):.3f} GB!"
                            )

                    with self.timer.time(step_name) as node:
                        X = step_fun.transform(X)
                        node["new_shape"] = {"rows": X.shape[0], "cols": X.shape[1]}
                        self.logger.debug(f"Finished data transformation of {step_fun.__class__.__name__}. New data size is {X.shape}")
        self.logger.info(f"Finished data transformation. New data size is {X.shape}")
        return X

    def get_pp_steps(self, X, y, metadata, **kwargs):
        idx_cat_col = np.where(metadata["categories"]["columns"])[0]
        idx_num_col = np.where(~np.array(metadata["categories"]["columns"]))[0]
        has_cat = len(idx_cat_col) > 0
        has_num = len(idx_num_col) > 0

        cat_steps = []
        num_steps = []
        treated_kws = []

        # step 1: imputation
        if np.any(pd.isnull(X)):
            cat_steps.append(("cat_imputer", SimpleImputer(strategy="most_frequent")))
            num_steps.append(("num_imputer", SimpleImputer(strategy="median")))

        # step 2: encoding of categorical attributes
        if has_cat:
            if KEY_CAT_ENCODER not in kwargs:
                raise ValueError(
                    f"{KEY_CAT_ENCODER} must be specified if the dataset has categorical attributes."
                )

            # Categorical features
            if kwargs[KEY_CAT_ENCODER] == "onehot":
                cat_encoder = OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                )
            elif kwargs[KEY_CAT_ENCODER] == "ordinal":
                cat_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            else:
                raise ValueError(
                    f"Unknown {KEY_CAT_ENCODER} technique {kwargs['cat_encoder']}"
                )
            cat_steps.append((KEY_CAT_ENCODER, cat_encoder))
        elif KEY_CAT_ENCODER in kwargs:
            msg = f"The value for {KEY_CAT_ENCODER} is set even though the data has no categorical attributes."\
                  " This may indicate an inefficiency, because different values may tried without having any effect."
            if self.raise_exception_on_unsuitable_preprocessor:
                msg += "\nYou can avoid that this situation generates an exception"\
                       "by setting `raise_exception_on_unsuitable_preprocessor=False`."
                raise ValueError(msg)
            self.logger.warning(msg)

        treated_kws.append(KEY_CAT_ENCODER)

        # initialize steps with the preliminary transformers
        transformers = []
        if has_cat and cat_steps:
            transformers.append(
                ("cat_transformations", Pipeline(cat_steps), idx_cat_col)
            )
        if has_num and num_steps:
            transformers.append(
                ("num_transformations", Pipeline(num_steps), idx_num_col)
            )
        steps = [
            (
                "pre_numeric_pp",
                ColumnTransformer(transformers=transformers, remainder="passthrough"),
            )
        ]

        # step 6: feature selector
        if KEY_FEATURESELECTOR in kwargs:
            fs_val = kwargs[KEY_FEATURESELECTOR]

            if fs_val == "selectp":
                # as we want to keep a minimum of 1 feature, we need to ensure that
                # percentile >= int(100 / X.shape[1]) + 1) which is the percentile corresponding to 1 feature
                featureselector = SelectPercentile(
                    percentile=max(self.selectp_percentile, int(100 / X.shape[1]) + 1)
                )
            elif fs_val == "none":
                featureselector = None
            else:
                raise ValueError(f"Unknown {KEY_FEATURESELECTOR} technique {fs_val}")
            if featureselector is not None:
                steps.append((KEY_FEATURESELECTOR, featureselector))
            treated_kws.append(KEY_FEATURESELECTOR)

        # step 3, feature generation
        if KEY_FEATUREGEN in kwargs:
            featuregen_val = kwargs[KEY_FEATUREGEN]
            if featuregen_val == "poly":
                # we use include_bias=False as most linear models have a `fit_intercept=True`
                # by default
                featuregen = PolynomialFeatures(
                    degree=self.poly_degree, include_bias=False
                )
            elif featuregen_val == "none":
                featuregen = None
            else:
                raise ValueError(f"Unknown {KEY_FEATUREGEN} technique {featuregen_val}")
            if featuregen is not None:
                steps.append((KEY_FEATUREGEN, featuregen))
            treated_kws.append(KEY_FEATUREGEN)

        # step 4: scaling
        if KEY_SCALER in kwargs:
            scaler_val = kwargs[KEY_SCALER]
            if scaler_val == "minmax":
                scaler = MinMaxScaler()
            elif scaler_val == "std":
                scaler = StandardScaler(with_mean=True, with_std=self.std_with_std)
            elif scaler_val == "none":
                scaler = None
            else:
                raise ValueError(f"Unknown {KEY_SCALER} technique {scaler_val}")
            if scaler is not None:
                steps.append((KEY_SCALER, scaler))
            treated_kws.append(KEY_SCALER)

        # step 5: featuremapper
        if KEY_FEATUREMAPPER in kwargs:
            featuremapper_val = kwargs[KEY_FEATUREMAPPER]
            if featuremapper_val == "kernel_pca":
                # kernel_pca_n_components is a float in [0, 1] that represents the ratio of
                # components we keep. we map it back to an integer.
                if self.kernel_pca_kernel == "linear":
                    featuremapper = PCA(n_components=min(X.shape[0], max(1, int(self.kernel_pca_n_components * X.shape[1]))))
                else:
                    featuremapper = KernelPCA(
                    kernel=self.kernel_pca_kernel,
                    n_components=max(1, int(self.kernel_pca_n_components * X.shape[1])),
                )
            elif featuremapper_val == "lda":
                featuremapper = LinearDiscriminantAnalysis()
            elif featuremapper_val == "fastica":
                featuremapper = FastICA()
            elif featuremapper_val == "ka_rbf":
                featuremapper = RBFSampler()
            elif featuremapper_val == "ka_nystroem":
                featuremapper = Nystroem()
            elif featuremapper_val == "agglomerator":
                # If enable, n_features**2 combinations
                featuremapper = FeatureAgglomeration()
            elif featuremapper_val == "none":
                featuremapper = None
            else:
                raise ValueError(
                    f"Unknown {KEY_FEATUREMAPPER} technique {featuremapper_val}"
                )
            if featuremapper is not None:
                steps.append((KEY_FEATUREMAPPER, featuremapper))
            treated_kws.append(KEY_FEATUREMAPPER)

        # sanity check
        untreated_kws = [k for k in kwargs if k not in treated_kws]
        if untreated_kws:
            raise ValueError(f"Untreated pre-processing kwargs: {untreated_kws}")

        # return trained pipeline
        return steps if steps else None

    def _fit(self, X, y, X_valid, y_valid, X_test, y_test, metadata):

        # first give the workflow a chance to transform the data, e.g., with data augmentation
        X, y = self._transform_train_data_prior_to_standard_preprocessing(X, y)

        # transform all the data and store them
        self.logger.info(f"Starting transformation of training data of size {X.shape}")
        X_train_transformed = self.transform(X=X, y=y, metadata=metadata, timer_suffix="_train").astype(np.float32)  # create + fit pp pipeline
        self.logger.info(f"Finished transformation of training data. New size is {X_train_transformed.shape}")
        self.logger.info(f"Starting transformation of validation data of size {X_valid.shape}")
        X_valid_transformed = self.transform(X_valid, y_valid, metadata, timer_suffix="_valid").astype(np.float32)
        self.logger.info(f"Finished transformation of validation data. New size is {X_valid_transformed.shape}")
        self.logger.info(f"Starting transformation of test data of size {X_test.shape}")
        X_test_transformed = self.transform(X_test, y_test, metadata, timer_suffix="_test").astype(np.float32)
        self.logger.info(f"Finished transformation of test data. New size is {X_test_transformed.shape}")
        
        # fit main model
        self.logger.info(f"Now fitting the main model.")
        self._fit_model_after_transformation(X_train_transformed, y, X_valid_transformed, y_valid, X_test_transformed, y_test, metadata)

    def _transform_train_data_prior_to_standard_preprocessing(self, X, y):
        return X, y  # by default, no alterations are made, of course. Overwrite this function to do so.

    def _fit_model_after_transformation(self, X, y, X_valid, y_valid, X_test, y_test, metadata):
        raise NotImplementedError

    def _predict(self, X):
        X = self.pp_pipeline.transform(X).astype(np.float32)
        return self._predict_after_transform(X)

    def _predict_after_transform(self, X):
        raise NotImplementedError

    def _predict_proba(self, X):
        X = self.pp_pipeline.transform(X).astype(np.float32)
        return self._predict_proba_after_transform(X)

    def _predict_proba_after_transform(self, X):
        raise NotImplementedError

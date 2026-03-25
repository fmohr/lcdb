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
    OrConjunction
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
from lcdb.builder._base import AnticipatedMemoryError

KEY_CAT_ENCODER = "cat_encoder"
KEY_SCALER = "scaler"
KEY_FEATUREGEN = "featuregen"
KEY_FEATUREMAPPER = "decomposition"
KEY_FEATURESELECTOR = "featureselector"

CONFIG_SPACE = ConfigurationSpace(
    name="standard_preprocessing",
    space={
        KEY_CAT_ENCODER: Categorical(
            KEY_CAT_ENCODER, ["none", "onehot", "ordinal"], default="ordinal" # ordinal is not semantically correct in nominal features but standard in deep learning and much faster than one-hot. Trees are agnostic
        ),
        KEY_SCALER: Categorical(KEY_SCALER, ["none", "minmax", "std"], default="none"),
        KEY_FEATUREGEN: Categorical(KEY_FEATUREGEN, ["none", "poly"], default="none"),
        "poly_degree": Constant("poly_degree", 2),#Integer("poly_degree", bounds=(2, 2), default=2),
        KEY_FEATUREMAPPER: Categorical(
            KEY_FEATUREMAPPER,
            [
                "none",
                "pca",
                "kernel_pca",
                "lda",
                "fastica",
                "ka_rbf",
                "ka_nystroem",
                "agglomerator",
            ],
            default="none",
        ),
        # defines how many features we want to have after the projection; relative number with semantic that depends on the projection technique (ratio of current features except for LDA where it is the ration of the possible features given the number of classes)
        "projection_features": Float(
            "projection_features", bounds=(0.01, 1.0), default=0.1 # by default reduce to 10% of the features
        ),

        # how many features to generate with the sample (RBFSampler or Nystroem Sampler)
        "feature_map_size": Integer(
            "feature_map_size", bounds=(1, 1000), default=1.0
        ),
        "kernel_mapper_kernel": Categorical(name="kernel_mapper_kernel", items=["cosine", "rbf", "poly", "sigmoid"]),
        "kernel_mapper_degree": Integer("kernel_mapper_degree", bounds=(2, 5), default=2),
        "kernel_mapper_coef0": Float("kernel_mapper_coef0", bounds=(0, 10**2), default=0, log=False),
        "kernel_mapper_gamma": Float("kernel_mapper_gamma", bounds=(10**-6, 10**6), default=1, log=True),
        KEY_FEATURESELECTOR: Categorical(
            KEY_FEATURESELECTOR,
            ["none", "selectp"],
            default="none",
        ),
        "selectp_percentile": Integer(
            "selectp_percentile", bounds=(1, 100), default=100
        ),
        "std_with_std": Categorical("std_with_std", [True, False], default=True)
    },
)

CONFIG_SPACE.add(
    [
        
        # only enable projection features if there is a feature projection method chosen
        InCondition(
            CONFIG_SPACE["projection_features"],
            CONFIG_SPACE[KEY_FEATUREMAPPER],
            ["kernel_pca", "lda", "fastica", "agglomerator"],
        ),
        
        # enable polynomial features only if there is no kernel feature mapper
        InCondition(
            CONFIG_SPACE[KEY_FEATUREGEN],
            CONFIG_SPACE[KEY_FEATUREMAPPER],
            ["none", "lda", "fastica", "agglomerator"],
        ),

        # only select kernel if we have a kernel method
        InCondition(
            CONFIG_SPACE["kernel_mapper_kernel"],
            CONFIG_SPACE[KEY_FEATUREMAPPER],
            ["kernel_pca", "ka_nystroem"]
        ),

        # only select degree and coef0 if we have a polynomial kernel
        InCondition(
            CONFIG_SPACE["kernel_mapper_degree"],
            CONFIG_SPACE["kernel_mapper_kernel"],
            ["poly", "sigmoid"]
        ),
        InCondition(
            CONFIG_SPACE["kernel_mapper_coef0"],
            CONFIG_SPACE["kernel_mapper_kernel"],
            ["poly","sigmoid"]
        ),

        # Gamma is set if we have a kernel based method (RBFSampler does not receive a kernel so requires its own enabling to make sure that we get values for this)
        OrConjunction(
            InCondition(
                CONFIG_SPACE["kernel_mapper_gamma"],
                CONFIG_SPACE["kernel_mapper_kernel"],
                ["poly","sigmoid", "rbf"]
            ),
            EqualsCondition(
                CONFIG_SPACE["kernel_mapper_gamma"],
                CONFIG_SPACE[KEY_FEATUREMAPPER],
                ["ka_rbf"]
            )
        ),

        InCondition(
            CONFIG_SPACE["feature_map_size"],
            CONFIG_SPACE[KEY_FEATUREMAPPER],
            ["ka_rbf", "ka_nystroem"],
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
        n_jobs=1,
        memory_limit_in_bytes=None,
        kernel_mapper_kernel="rbf",
        kernel_mapper_degree=2,
        kernel_mapper_coef0=0,
        kernel_mapper_gamma=1.0,
        projection_features=1.0,
        feature_map_size=1000,
        selectp_percentile=100,
        poly_degree=2,
        std_with_std=True,
        raise_exception_on_unsuitable_preprocessor=True,
        **kwargs,
    ):
        super().__init__(
            timer=timer,
            logger=logger,
            random_state=random_state,
            memory_limit_in_bytes=memory_limit_in_bytes,
            n_jobs=n_jobs
        )

        # extract preprocessing hyperparameters
        self.pp_kws = kwargs
        self.pp_pipeline = None

        self.kernel_mapper_kernel = kernel_mapper_kernel
        self.kernel_mapper_poly_degree = 3 if np.isnan(kernel_mapper_degree) else int(kernel_mapper_degree)
        self.kernel_mapper_coef0 = 1.0 if np.isnan(kernel_mapper_coef0) else kernel_mapper_coef0
        self.kernel_mapper_gamma = None if np.isnan(kernel_mapper_gamma) else kernel_mapper_gamma
        self.projection_features = projection_features
        self.feature_map_size = None if np.isnan(feature_map_size) else int(feature_map_size)
        self.selectp_percentile = selectp_percentile
        self.poly_degree = None if np.isnan(poly_degree) else int(poly_degree)
        self.std_with_std = std_with_std
        self.raise_exception_on_unsuitable_preprocessor = raise_exception_on_unsuitable_preprocessor
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        if 'is_randomizable' in cls.__dict__:
            raise TypeError("is_randomizable cannot be overridden for sub-workflows of PreprocessedWorkflow, because this is always and necessarily randomizable. ")

    @classmethod
    def is_randomizable(cls):
        return True

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
        
        # in the LDA we use SVD, which requires adds the computation of a min(n, d) square matrix
        if isinstance(pre_processor, LinearDiscriminantAnalysis):
            s = min(input_shape[0], input_shape[1])
            return estimate_memory_consumption_for_dataset((s, s))
        
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
                    self.logger.info(f"Expected memory usage run pre-processor {step_fun.__class__.__name__}: {used_memory_bytes / (1024**3):.3f}GB.")
                    if used_memory_bytes > self.memory_limit_in_bytes:
                        raise AnticipatedMemoryError(
                            f"{step_name} ({step_fun.__class__.__name__}) is predicted to consume approximately {used_memory_bytes/(1024**3):.3f} GB. "
                            f"The permitted maximum is {self.memory_limit_in_bytes/(1024**3):.3f} GB!"
                            )

                    # transform the data
                    with self.timer.time(step_name) as node:
                        self.logger.debug(f"Applying fit_transform of {step_name} ({step_fun}) to data of shape {X.shape}")
                        if step_name != "pre_numeric_pp":
                            assert not pd.isna(X).any(), f"there are still nan values in the input when applying {step_fun} as {step_name}"
                            assert not pd.isna(X).any(), f"there are still inf values in the input when applying {step_fun} as {step_name}"
                        X = step_fun.fit_transform(X, y=y)

                        node["new_shape"] = {"rows": X.shape[0], "cols": X.shape[1]}
                        self.logger.debug(f"New data shape is {X.shape}")
                self.pp_pipeline = Pipeline(steps=pp_steps)
            else:
                self.logger.debug(f"Starting data transformation with input size {X.shape}")
                for step_name, step_fun in self.pp_pipeline.steps:

                    # anticipate memory usage and possibly avoid execution
                    used_memory_bytes = self._anticipate_required_memory_for_transform(X.shape, step_fun)
                    self.logger.info(f"Expected memory usage run pre-processor {step_fun.__class__.__name__}: {used_memory_bytes / (1024**3):.3f}GB.")
                    if used_memory_bytes > self.memory_limit_in_bytes:
                        raise AnticipatedMemoryError(
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
        if type(metadata["categories"]["columns"]) != np.ndarray or len(metadata["categories"]["columns"].shape) != 1:
            raise ValueError(f"The binary mask for categorical columns must be a binary 1D numpy array but is {type(metadata['categories']['columns'])}")
        idx_cat_col = np.where(metadata["categories"]["columns"])[0]
        idx_num_col = np.where(~np.array(metadata["categories"]["columns"]))[0]
        has_cat = len(idx_cat_col) > 0
        has_num = len(idx_num_col) > 0

        cat_steps = []
        num_steps = []
        treated_kws = []

        # step 1: always set an imputer (even if no missing values are present; this is because there might be missing values in the validation/test data even if there were none in the training data)
        cat_steps.append(("cat_imputer", SimpleImputer(strategy="most_frequent")))
        num_steps.append(("num_imputer", SimpleImputer(strategy="median")))

        # step 2: encoding of categorical attributes
        if has_cat:
            self.logger.info(f"Data has categorical attributes. Encoder to treat these is {kwargs[KEY_CAT_ENCODER] if KEY_CAT_ENCODER in kwargs else 'not specified'}.")
            if KEY_CAT_ENCODER not in kwargs or kwargs[KEY_CAT_ENCODER] == "none":
                msg = f"The value for {KEY_CAT_ENCODER} is set to none even though the data has categorical attributes."
                if self.raise_exception_on_unsuitable_preprocessor:
                    raise ValueError(msg)
                else:
                    msg += " Switching to ordinal encoding."
                    self.logger.warning(msg)
                    kwargs[KEY_CAT_ENCODER] = "ordinal"

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
        else:
            self.logger.info("Data has no categorical attributes.")
            if KEY_CAT_ENCODER in kwargs and kwargs["cat_encoder"] != "none":
                msg = f"The value for {KEY_CAT_ENCODER} is set (to {kwargs['cat_encoder']}) even though the data has no categorical attributes."\
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
            self.logger.debug(f"Registering categorical transformations for categorical columns {idx_cat_col}.")
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

        # simulate this encoding
        num_features_after_categorical_encoding = Pipeline(steps).fit_transform(X, y).shape[1]
        if num_features_after_categorical_encoding != X.shape[1]:
            self.logger.info(f"Categorical encoding will change number of features from {X.shape[1]} to {num_features_after_categorical_encoding}")

        # step 2: feature selector
        num_features_after_feature_selector = num_features_after_categorical_encoding
        if KEY_FEATURESELECTOR in kwargs:
            fs_val = kwargs[KEY_FEATURESELECTOR]

            if fs_val == "selectp":
                # as we want to keep a minimum of 1 feature, we need to ensure that
                # percentile >= int(100 / X.shape[1]) + 1) which is the percentile corresponding to 1 feature
                percentile = max(self.selectp_percentile, int(100 / num_features_after_categorical_encoding) + 1)
                num_fractional_features_after_feature_selector = num_features_after_categorical_encoding * percentile / 100
                num_features_after_feature_selector = int(np.round(num_fractional_features_after_feature_selector))  # enforce round up at exactly x.5
                if num_features_after_feature_selector == 1 and KEY_FEATUREMAPPER in kwargs and kwargs[KEY_FEATUREMAPPER] != "none":
                    percentile_new = max(1, int(100 * 2.01 / num_features_after_categorical_encoding))
                    self.logger.warning(f"Modifying percentile slightly from {percentile} to {percentile_new} so that at least two features survive because otherwise there would be only one, and there is a feature mapper active.")
                    percentile = percentile_new
                    num_fractional_features_after_feature_selector = num_features_after_categorical_encoding * percentile / 100
                    num_features_after_feature_selector = int(np.round(num_fractional_features_after_feature_selector))  # enforce round up at exactly x.5
                featureselector = SelectPercentile(percentile=percentile)
                self.logger.debug(f"Feature selector would use percentile {percentile} and reduce number of features to {num_fractional_features_after_feature_selector}")
            elif fs_val == "none":
                featureselector = None
            else:
                raise ValueError(f"Unknown {KEY_FEATURESELECTOR} technique {fs_val}")
            if featureselector is not None:
                steps.append((KEY_FEATURESELECTOR, featureselector))
            treated_kws.append(KEY_FEATURESELECTOR)
        if num_features_after_feature_selector != num_features_after_categorical_encoding:
            self.logger.info(f"Feature selection will change number of features from {num_features_after_categorical_encoding} to {num_features_after_feature_selector}")

        # step 3, feature generation
        num_features_after_feature_generation = num_features_after_feature_selector
        if KEY_FEATUREGEN in kwargs:
            featuregen_val = kwargs[KEY_FEATUREGEN]
            if featuregen_val == "poly":
                # we use include_bias=False as most linear models have a `fit_intercept=True`
                # by default
                featuregen = PolynomialFeatures(
                    degree=self.poly_degree, include_bias=False
                )
                num_features_after_feature_generation =  featuregen._num_combinations(
                    n_features=num_features_after_feature_selector,
                    min_degree=0,
                    max_degree=featuregen.degree,
                    interaction_only=featuregen.interaction_only,
                    include_bias=featuregen.include_bias,
                )
            elif featuregen_val == "none":
                featuregen = None
            else:
                raise ValueError(f"Unknown {KEY_FEATUREGEN} technique {featuregen_val}")
            if featuregen is not None:
                steps.append((KEY_FEATUREGEN, featuregen))
            treated_kws.append(KEY_FEATUREGEN)
        
        if num_features_after_feature_generation != num_features_after_feature_selector:
            self.logger.info(f"Feature generation will change number of features from {num_features_after_feature_selector} to {num_features_after_feature_generation}")

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
            if featuremapper_val != "none":
                if featuremapper_val in ["pca", "kernel_pca", "fastica", "agglomerator"]:
                    
                    n_components = max(1, int(self.projection_features * num_features_after_feature_generation))
                    # the number of features to reduce to is a float in [0, 1] that represents the ratio of components we keep.
                        # the problem is:
                        #   - if the number is close to 0, then we are eliminating all the features (motivating to move to the absolute number of components).
                        #   - if we create a mapping to absolute values, we do not know exactly how many features will be left (motivating a relative number of components).
                        # solution: we pre-compute the shape of the data after previous pre-processing steps and then employ the mapping to integer level.

                    if featuremapper_val == "pca":
                        n_components = min(X.shape[0], n_components)
                        featuremapper = PCA(
                            n_components=n_components,
                            random_state=self.random_state
                        )

                    if featuremapper_val == "kernel_pca":
                        n_components = min(X.shape[0], n_components)
                        featuremapper = KernelPCA(
                            kernel=self.kernel_mapper_kernel,
                            degree=self.kernel_mapper_poly_degree, # automatically ignored if kernel is not polynomial
                            coef0=self.kernel_mapper_coef0,
                            gamma=self.kernel_mapper_gamma,
                            n_components=n_components,
                            random_state=self.random_state
                        )
                    elif featuremapper_val == "fastica":
                        n_components = min(X.shape[0], n_components)
                        featuremapper = FastICA(random_state=self.random_state)
                    elif featuremapper_val == "agglomerator":
                        # If enable, n_features**2 combinations
                        featuremapper = FeatureAgglomeration(n_clusters=n_components)
                
                elif featuremapper_val == "lda":
                    # in LDA we will use the number as the ratio of *classes*
                    num_possible_features = min(len(self.infos["classes_train_orig"]) - 1, num_features_after_feature_generation)
                    n_components = max(1, int(self.projection_features * num_possible_features))
                    featuremapper = LinearDiscriminantAnalysis(n_components=n_components)
                elif featuremapper_val == "ka_rbf":
                    n_components = int(self.feature_map_size)
                    featuremapper = RBFSampler(
                        gamma=self.kernel_mapper_gamma,
                        n_components=n_components,
                        random_state=self.random_state
                    )
                elif featuremapper_val == "ka_nystroem":
                    n_components = int(self.feature_map_size)
                    featuremapper = Nystroem(
                        kernel=self.kernel_mapper_kernel,
                        degree=self.kernel_mapper_poly_degree, # automatically ignored if kernel is not polynomial
                        coef0=self.kernel_mapper_coef0,
                        gamma=self.kernel_mapper_gamma,
                        n_components=n_components,
                        random_state=self.random_state
                    )
                else:
                    raise ValueError(
                        f"Unknown {KEY_FEATUREMAPPER} technique {featuremapper_val}"
                    )
                
                # inform about change in the number of columns
                if n_components != num_features_after_feature_generation:
                    self.logger.info(f"Feature mapper {featuremapper.__class__.__name__} will change number of features from {num_features_after_feature_generation} to {n_components}")
            else:
                featuremapper = None
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
        self.logger.info(f"Starting transformation of training data of size {X.shape}.")
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

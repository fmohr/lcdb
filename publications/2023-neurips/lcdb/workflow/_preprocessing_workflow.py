import warnings

import numpy as np
import pandas as pd
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
    EqualsCondition,
    InCondition,
)
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from ._base_workflow import BaseWorkflow

KEY_CAT_ENCODER = "cat_encoder"
KEY_SCALER = "scaler"
KEY_FEATUREGEN = "featuregen"
KEY_FEATUREMAPPER = "decomposition"
KEY_FEATURESELECTOR = "featureselector"

CONFIG_SPACE = ConfigurationSpace(
    name="standard_preprocessing",
    space={
        KEY_CAT_ENCODER: Categorical(
            KEY_CAT_ENCODER, ["onehot", "ordinal"], default="ordinal"
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
            ["none", "selectp", "generic"],
            default="generic",
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
        "poly_degree": Integer("poly_degree", bounds=(2, 5), default=2),
        "std_with_std": Categorical("std_with_std", [True, False], default=True),
    },
)

CONFIG_SPACE.add_conditions(
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


class PreprocessedWorkflow(BaseWorkflow):
    _config_space = CONFIG_SPACE

    def __init__(
        self,
        timer=None,
        kernel_pca_kernel="linear",
        kernel_pca_n_components=1.0,
        selectp_percentile=100,
        poly_degree=2,
        std_with_std=True,
        **kwargs,
    ):
        super().__init__(timer)

        # extract preprocessing hyperparameters
        self.pp_kws = kwargs
        self.pp_pipeline = None

        self.kernel_pca_kernel = kernel_pca_kernel
        self.kernel_pca_n_components = kernel_pca_n_components
        self.selectp_percentile = selectp_percentile
        self.poly_degree = poly_degree
        self.std_with_std = std_with_std

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

    def _transform(self, X, y, metadata):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            if not self.transform_fitted:
                self.pp_pipeline = self.get_pp_pipeline(X, y, metadata, **self.pp_kws)
            X = self.pp_pipeline.transform(X) if self.pp_pipeline is not None else X

        return X

    def get_pp_pipeline(self, X, y, metadata, **kwargs):
        idx_cat_col = np.where(metadata["categories"]["columns"])[0]
        idx_num_col = np.where(~metadata["categories"]["columns"])[0]
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

            # print(metadata["num_classes"], featuremapper_val)
            # if metadata["num_classes"] == 2 and featuremapper_val == "lda":
            #     fs_val = "none"

            if fs_val == "selectp":
                # as we want to keep a minimum of 1 feature, we need to ensure that
                # percentile >= int(100 / X.shape[1]) + 1) which is the percentile corresponding to 1 feature
                featureselector = SelectPercentile(
                    percentile=max(self.selectp_percentile, int(100 / X.shape[1]) + 1)
                )
            elif fs_val == "generic":
                featureselector = GenericUnivariateSelect()
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
                featuregen = PolynomialFeatures(degree=self.poly_degree)
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
        untreated_kws = [k for k in kwargs if not k in treated_kws]
        if untreated_kws:
            raise ValueError(f"Untreated pre-processing kwargs: {untreated_kws}")

        # return trained pipeline
        return Pipeline(steps).fit(X, y) if steps else None

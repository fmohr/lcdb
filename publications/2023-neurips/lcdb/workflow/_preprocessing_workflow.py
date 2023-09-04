from abc import ABC

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile
from sklearn.cluster import FeatureAgglomeration
from sklearn.pipeline import Pipeline
from ._base_workflow import BaseWorkflow
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from ConfigSpace import (
    Categorical,
    ConfigurationSpace
)

KEY_CAT_ENCODER = "pp_cat_encoder"
KEY_SCALER = "pp_scaler"
KEY_FEATUREGEN = "pp_featuregen"
KEY_FEATUREMAPPER = "pp_decomposition"
KEY_FEATURESELECTOR = "pp_featureselector"

CONFIG_SPACE = ConfigurationSpace(
    name="standard_preprocessing",
    space={
        KEY_CAT_ENCODER: Categorical(
            KEY_CAT_ENCODER, ["onehot", "ordinal"], default="ordinal"
        ),
        KEY_SCALER: Categorical(
            KEY_SCALER, ["none", "minmax", "std"], default="none"
        ),
        KEY_FEATUREGEN: Categorical(
            KEY_FEATUREGEN, ["none", "poly2", "poly3"], default="none"
        ),
        KEY_FEATUREMAPPER: Categorical(
            KEY_FEATUREMAPPER, ["none", "pca", "kernelpca", "lda", "fastica", "ka_rbf", "ka_nystroem", "agglomerator"], default="none"
        ),
        KEY_FEATURESELECTOR: Categorical(
            KEY_FEATURESELECTOR, ["none", "select50", "select75", "select90", "generic"], default="generic"
        )
    }
)


class PreprocessedWorkflow(BaseWorkflow, ABC):
    _config_space = CONFIG_SPACE

    def __init__(self, **kwargs):
        super().__init__()
        
        # extract preprocessing hyperparameters
        self.pp_kws = {key: val for key, val in kwargs.items() if key[:3] == "pp_"}
        self.pp_pipeline = None

    @classmethod
    def config_space(cls, techniques=[KEY_CAT_ENCODER, KEY_SCALER, KEY_FEATUREGEN, KEY_FEATUREMAPPER, KEY_FEATURESELECTOR]):
        cs = ConfigurationSpace()
        hp_names = set(hp.name for hp in cls._config_space.get_hyperparameters())
        unknown_techniques = set(techniques).difference(hp_names)
        if unknown_techniques:
            raise ValueError(f"Unknown preprocessing technique keys: {unknown_techniques}")
        for hp in cls._config_space.get_hyperparameters():
            if hp.name in techniques:
                cs.add_hyperparameter(hp)
        return cs

    def _transform(self, X, y, metadata):
        if not self.transform_fitted:
            self.pp_pipeline = self.get_pp_pipeline(X, y, metadata, **self.pp_kws)
        return self.pp_pipeline.transform(X) if self.pp_pipeline is not None else X

    @staticmethod
    def get_pp_pipeline(X, y, metadata, **kwargs):

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
                raise ValueError(f"{KEY_CAT_ENCODER} must be specified if the dataset has categorical attributes.")

            # Categorical features
            if kwargs[KEY_CAT_ENCODER] == "onehot":
                cat_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            elif kwargs[KEY_CAT_ENCODER] == "ordinal":
                cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            else:
                raise ValueError(
                    f"Unknown {KEY_CAT_ENCODER} technique {kwargs['cat_encoder']}"
                )
            cat_steps.append((KEY_CAT_ENCODER, cat_encoder))
            treated_kws.append(KEY_CAT_ENCODER)

        # initialize steps with the preliminary transformers
        transformers = []
        if has_cat and cat_steps:
            transformers.append(("cat_transformations", Pipeline(cat_steps), idx_cat_col))
        if has_num and num_steps:
            transformers.append(("num_transformations", Pipeline(num_steps), idx_num_col))
        steps = [
            ("pre_numeric_pp", ColumnTransformer(transformers=transformers, remainder="passthrough"))
        ]

        # step 3, feature generation
        if KEY_FEATUREGEN in kwargs:
            featuregen_val = kwargs[KEY_FEATUREGEN]
            if featuregen_val == "poly2":
                featuregen = PolynomialFeatures(degree=2)
            elif featuregen_val == "poly3":
                featuregen = PolynomialFeatures(degree=3)
            elif featuregen_val == "none":
                featuregen = None
            else:
                raise ValueError(
                    f"Unknown {KEY_FEATUREGEN} technique {featuregen_val}"
                )
            if featuregen is not None:
                steps.append((KEY_FEATUREGEN, featuregen))
            treated_kws.append(KEY_FEATUREGEN)

        # step 4: scaling
        if KEY_SCALER in kwargs:
            scaler_val = kwargs[KEY_SCALER]
            if scaler_val == "minmax":
                scaler = MinMaxScaler()
            elif scaler_val == "std":
                scaler = StandardScaler()
            elif scaler_val == "none":
                scaler = None
            else:
                raise ValueError(
                    f"Unknown {KEY_SCALER} technique {scaler_val}"
                )
            if scaler is not None:
                steps.append((KEY_SCALER, scaler))
            treated_kws.append(KEY_SCALER)

        # step 5: featuremapper
        if KEY_FEATUREMAPPER in kwargs:
            featuremapper_val = kwargs[KEY_FEATUREMAPPER]
            if featuremapper_val == "pca":
                featuremapper = PCA()
            elif featuremapper_val == "kernelpca":
                featuremapper = KernelPCA()
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

        # step 6: feature selector
        if KEY_FEATURESELECTOR in kwargs:
            fs_val = kwargs[KEY_FEATURESELECTOR]
            if fs_val == "select50":
                featureselector = SelectPercentile(percentile=50)
            elif fs_val == "select75":
                featureselector = SelectPercentile(percentile=75)
            elif fs_val == "select90":
                featureselector = SelectPercentile(percentile=90)
            elif fs_val == "generic":
                featureselector = GenericUnivariateSelect()
            elif fs_val == "none":
                featureselector = None
            else:
                raise ValueError(
                    f"Unknown {KEY_FEATURESELECTOR} technique {fs_val}"
                )
            if featureselector is not None:
                steps.append((KEY_FEATURESELECTOR, featureselector))
            treated_kws.append(KEY_FEATURESELECTOR)

        # sanity check
        untreated_kws = [k for k in kwargs if not k in treated_kws]
        if untreated_kws:
            raise ValueError(f"Untreated pre-processing kwargs: {untreated_kws}")

        # return trained pipeline
        return Pipeline(steps).fit(X, y) if steps else None

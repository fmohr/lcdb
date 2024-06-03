import sklearn
from ConfigSpace import (
    Constant,
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
)
from lcdb.scorer import ClassificationScorer
from sklearn.ensemble import ExtraTreesClassifier
from lcdb.utils import get_schedule

import numpy as np

from ...utils import filter_keys_with_prefix
from ._forest import ForestWorkflow

CONFIG_SPACE = ConfigurationSpace(
    name="sklearn.RandomForestWorkflow",
    space={},
)


class ExtraTreesWorkflow(ForestWorkflow):
    # Static Attribute
    _config_space = CONFIG_SPACE
    _config_space.add_configuration_space(
        prefix="",
        delimiter="",
        configuration_space=ForestWorkflow.config_space(),
    )

    def __init__(
        self,
        timer=None,
        n_estimators=1,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        random_state=None,
        **kwargs,
    ):

        learner_kwargs = dict(
            n_estimators=1,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            warm_start=True,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            random_state=random_state,
        )

        super().__init__(
            forest=ExtraTreesClassifier(**learner_kwargs),
            n_estimators=n_estimators,
            timer=timer,
            **filter_keys_with_prefix(kwargs, prefix="pp@")
        )

    @classmethod
    def config_space(cls):
        return cls._config_space

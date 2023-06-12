from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform
from ConfigSpace.read_and_write import json as cs_json

# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
# most likely want to restrict to booster="gbtree" and tree_method="exact"

# https://proceedings.mlr.press/v188/makarova22a/makarova22a.pdf

cs = ConfigurationSpace(
    name="xgboost",
    seed=1234,
    space={
        # "n_estimators": Integer("n_estimators, bounds=(2, 2**9), distribution=Uniform(), log=True), fidelity parameter we handle manually
        "learning_rate": Float(
            "learning_rate",
            bounds=(10**-6, 1),
            distribution=Uniform(),
            default=0.3,
            log=True,
        ),
        "gamma": Float(
            "gamma",
            bounds=(10**-6, 2**6),
            distribution=Uniform(),
            default=10**-6,
            log=True,
        ),  # normal default would be 0
        "min_child_weight": Float(
            "min_child_weight",
            bounds=(10**-6, 2**5),
            distribution=Uniform(),
            default=1,
            log=True,
        ),
        "max_depth": Integer(
            "max_depth", bounds=(2, 2**5), distribution=Uniform(), default=6, log=True
        ),
        "subsample": Float(
            "subsample", bounds=(0.5, 1), distribution=Uniform(), default=1
        ),
        "colsample_bytree": Float(
            "colsample_bytree", bounds=(0.3, 1), distribution=Uniform(), default=1
        ),
        "reg_alpha": Float(
            "reg_alpha",
            bounds=(10**-6, 2),
            distribution=Uniform(),
            default=10**-6,
            log=True,
        ),  # normal default would be 0
        "reg_lambda": Float(
            "reg_lambda",
            bounds=(10**-6, 2),
            distribution=Uniform(),
            default=1,
            log=True,
        ),
    },
)

# cs_string = cs_json.write(cs)
# with open("configspace_xgboost.json", "w") as f:
#    f.write(cs_string)

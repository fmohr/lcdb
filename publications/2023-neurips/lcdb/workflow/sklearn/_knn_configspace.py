from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform, Constant, Categorical, EqualsCondition, OrConjunction
from ConfigSpace.read_and_write import json as cs_json

def get_configspace():

    cs = ConfigurationSpace(
        name="knn",
        space={
            "n_neighbors": Integer("n_neighbors", (1, 1000), default=5, log=True),
            "scaler": Categorical("scaler", ["minmax", "standardize", "none"], default="none"),
            "weights": Categorical("weights", ["uniform", "distance"], default="uniform"),
            "p": Integer("p", (1, 2), default=2),
        }
    )
    return cs


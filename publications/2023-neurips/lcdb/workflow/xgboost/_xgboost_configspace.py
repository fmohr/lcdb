from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform, Categorical

def get_configspace():

  cs = ConfigurationSpace(
      name="xgboost",
      space={
          # FIXME: for now include n_estimators in the configspace, but we should handle it manually as fidelity parameter
          "n_estimators": Integer("n_estimators", bounds=(2, 2**9), distribution=Uniform(), log=True),
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
          "scaler": Categorical("scaler", ["minmax", "standardize", "none"], default="none"),
      },
    )
  return cs
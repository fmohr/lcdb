from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform, Constant, Categorical, EqualsCondition, OrConjunction
from ConfigSpace.read_and_write import json as cs_json

cs = ConfigurationSpace(
    name="liblinear",
    space={
        "dual": Categorical("dual", [False, True], default=True),
        "C": Float("C", bounds=(1e-12, 1e12), distribution=Uniform(), default=1, log=True),
        "multiclass": Categorical("multiclass", ["ovr", "ovo-scikit"], default="ovr"),
        "tol": Float("tol", bounds=(1e-5, 2), distribution=Uniform(), default=1e-3, log=True),
        "max_iter": Integer("max_iter", bounds=(100, 1000000), log=True),
        "class_weight": Categorical("class_weight", ['balanced', 'none']),
    }
)

loss = Categorical("loss", ["hinge", "squared_hinge"], default="squared_hinge")
penalty = Categorical("penalty", ["l2", "l1"], default="l2")
cs.add_hyperparameters([loss, penalty])

fit_intercept = Categorical("fit_intercept", [False, True], default=True)
intercept_scaling = Float("intercept_scaling", bounds=(1.0, 1e3), default=1.0, log=True)
cs.add_hyperparameters([fit_intercept, intercept_scaling])

cond_1 = EqualsCondition(loss, penalty, 'l2')
cond_2 = EqualsCondition(intercept_scaling, fit_intercept, True)
cs.add_conditions([cond_1, cond_2])

cs_string = cs_json.write(cs)
with open("_liblinear_cs.json", "w") as f:
    f.write(cs_string)

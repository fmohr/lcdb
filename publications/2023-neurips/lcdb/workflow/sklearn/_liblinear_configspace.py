from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform, Constant, Categorical, EqualsCondition, \
    OrConjunction, ForbiddenEqualsClause, ForbiddenAndConjunction
from ConfigSpace.read_and_write import json as cs_json

def get_configspace():

    cs = ConfigurationSpace(
        name="liblinear",
        space={
            "dual": Categorical("dual", [False, True], default=True),
            "C": Float("C", bounds=(1e-12, 1e12), distribution=Uniform(), default=1, log=True),
            "multiclass": Categorical("multiclass", ["ovr", "ovo-scikit"], default="ovr"),
            "tol": Float("tol", bounds=(1e-5, 2), distribution=Uniform(), default=1e-3, log=True),
            "max_iter": Integer("max_iter", bounds=(100, 1000000), log=True),
            "class_weight": Categorical("class_weight", ['balanced', 'none']),
            "loss": Categorical("loss", ["hinge", "squared_hinge"], default="squared_hinge"),
            "penalty": Categorical("penalty", ["l2", "l1"], default="l2"),
            "scaler": Categorical("scaler", ["minmax", "standardize", "none"], weights=[0.5, 0.5, 0.0], default="none"),
        }
    )

    fit_intercept = Categorical("fit_intercept", [False, True], default=True)
    intercept_scaling = Float("intercept_scaling", bounds=(1.0, 1e3), default=1.0, log=True)
    cs.add_hyperparameters([fit_intercept, intercept_scaling])

    cond_2 = EqualsCondition(intercept_scaling, fit_intercept, True)
    cs.add_conditions([cond_2])

    forbidden_clause_a = ForbiddenEqualsClause(cs["loss"], "hinge")
    forbidden_clause_b = ForbiddenEqualsClause(cs["penalty"], "l1")

    forbidden_clause = ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_b)
    cs.add_forbidden_clause(forbidden_clause)

    forbidden_clause_c = ForbiddenEqualsClause(cs["loss"], "squared_hinge")
    forbidden_clause_d = ForbiddenEqualsClause(cs["penalty"], "l1")
    forbidden_clause_e = ForbiddenEqualsClause(cs["dual"], True)

    forbidden_clause2 = ForbiddenAndConjunction(forbidden_clause_c, forbidden_clause_d)
    forbidden_clause3 = ForbiddenAndConjunction(forbidden_clause2, forbidden_clause_e)
    cs.add_forbidden_clause(forbidden_clause3)

    forbidden_clause_f = ForbiddenEqualsClause(cs["loss"], "hinge")
    forbidden_clause_g = ForbiddenEqualsClause(cs["penalty"], "l2")
    forbidden_clause_h = ForbiddenEqualsClause(cs["dual"], False)

    forbidden_clause4 = ForbiddenAndConjunction(forbidden_clause_f, forbidden_clause_g)
    forbidden_clause5 = ForbiddenAndConjunction(forbidden_clause4, forbidden_clause_h)
    cs.add_forbidden_clause(forbidden_clause5)

    return cs

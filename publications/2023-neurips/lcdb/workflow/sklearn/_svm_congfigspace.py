from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform, Constant, Categorical, EqualsCondition, OrConjunction
from ConfigSpace.read_and_write import json as cs_json

cs = ConfigurationSpace(
    name="svm",
    space={
        "C": Float(
            "C",
            bounds=(10**(-12), 10**12),
            distribution=Uniform(),
            default=1,
            log=True,
        ),
        # "shrinking": Constant("shrinking", True), # this is not allowed for some reason
        # "probability": Constant("probability", False),
        "break_ties": Categorical("break_ties", [False, True], default=False),
        "tol": Constant("tol", 1e-3)
    },
)

kernel = Categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], default='linear')
gamma = Float(
            "gamma",
            bounds=(10**(-12), 10**12),
            distribution=Uniform(),
            default=1,
            log=True,
        )
degree = Integer(
            "degree",
            bounds=(2, 5),
            distribution=Uniform(),
            default=2,
            log=False,
        )
coef0 = Float(
            "coef0",
            bounds=(10**(-12), 10**2),
            distribution=Uniform(),
            default=0.1, # strange, 0.0 is not allowed?
            log=True
        )
cs.add_hyperparameters([kernel, degree, coef0, gamma])

# degree
cond_1 = EqualsCondition(degree, kernel, 'poly')

# coef0
cond_2 = OrConjunction(
    EqualsCondition(coef0, kernel, 'poly'),
    EqualsCondition(coef0, kernel, 'sigmoid')
)

# gamma
cond_3 = OrConjunction(
    EqualsCondition(gamma, kernel, 'rbf'),
    EqualsCondition(gamma, kernel, 'poly'),
    EqualsCondition(gamma, kernel, 'sigmoid')
)
cs.add_conditions([cond_1, cond_2, cond_3])

cs_string = cs_json.write(cs)
with open("_svm_cs.json", "w") as f:
    f.write(cs_string)

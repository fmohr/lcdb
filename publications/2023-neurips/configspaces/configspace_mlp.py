from ConfigSpace import ConfigurationSpace, Integer, Float, Uniform
from ConfigSpace.read_and_write import json as cs_json

# https://github.com/automl/Auto-PyTorch
# funnel shaped MLP with SGD with cosine annealing without restarts

# https://www.computer.org/csdl/journal/tp/2021/09/09382913/1saYy4aNXwY
# https://github.com/automl/LCBench
# defaults?

cs = ConfigurationSpace(
    name="mlp",
    seed=1234,
    space={
        # "epoch": Integer("epoch, bounds=(1, 50), distribution=Uniform()), fidelity parameter we handle manually
        "batch_size": Integer(
            "batch_size", bounds=(2**4, 2**9), distribution=Uniform(), log=True
        ),
        "learning_rate": Float(
            "learning_rate",
            bounds=(10**-4, 10**-1),
            distribution=Uniform(),
            log=True,
        ),
        "momentum": Float("momentum", bounds=(0.1, 0.99), distribution=Uniform()),
        "weight_decay": Float(
            "weight_decay", bounds=(10**-5, 10**-1), distribution=Uniform()
        ),
        "num_layers": Integer("num_layers", bounds=(1, 5), distribution=Uniform()),
        "max_units": Integer(
            "max_units", bounds=(2**6, 2**10), distribution=Uniform(), log=True
        ),
        "max_dropout": Float("max_dropout", bounds=(0, 1), distribution=Uniform()),
    },
)

# cs_string = cs_json.write(cs)
# with open("configspace_mlp.json", "w") as f:
#    f.write(cs_string)

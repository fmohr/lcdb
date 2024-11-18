import os

import deephyper_benchmark as dhb

dhb.load("LCDB/hyperparameter_optimization")

from deephyper_benchmark.lib.lcdb.hyperparameter_optimization import hpo

problem = hpo.problem
run = hpo.run

if __name__ == "__main__":
    print(problem)

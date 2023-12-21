import os

if os.environ.get("DEEPHYPER_BENCHMARK_TASK") is None:
    os.environ["DEEPHYPER_BENCHMARK_TASK"] = "navalpropulsion"

import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular")

from deephyper_benchmark.lib.hpobench.tabular import hpo

problem = hpo.problem
run = hpo.run

if __name__ == "__main__":
    print(problem)
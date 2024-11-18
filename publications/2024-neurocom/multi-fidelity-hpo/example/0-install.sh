#!/bin/bash
# Before running this script, create a new conda environment
# $ conda create -n oneEpoch python=3.11 -y
# and then activate it
# $ conda activate oneEpoch

mkdir -p build && cd build/

# To install `deephyper`: the software providing hyperparameter 
# optimization and early discarding methods.
pip install "deephyper[hps,jax-cpu] @ git+https://github.com/deephyper/deephyper@develop"

# To install `deephyper_benchmark`: the software providing learning 
# curve benchmarks for hyperparameter optimization.
pip install -e "git+https://github.com/deephyper/benchmark.git@main#egg=deephyper-benchmark"

# To install the `HPOBench/tabular` (the 4 regression problems) with `deephyper_benchmark`
python -c "import deephyper_benchmark as dhb; dhb.install('HPOBench/tabular');"

# To install `dhexp`: the sofware developed to automate our experiments.
pip install -e "../../"
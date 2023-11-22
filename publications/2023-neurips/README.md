# LCDB 2.0: Learning Curves Database of Configurable Learning Workflows

LCDB 2.0 is a database of learning workflows with diverse configurations (or hyperparameters).

## Installation

The LCDB Package **alone** can be installed with:
```console
pip install -e "."
```

To install DeepHyper (develop) version:

```console
pip install "deephyper @ git+https://github.com/deephyper/deephyper@develop"
```

### ALCF - Polaris

```console
git clone -b deephyper git@github.com:fmohr/lcdb.git
mkdir -p lcdb/publications/2023-neurips/build
cd lcdb/publications/2023-neurips/build/
../install/polaris.sh
source activate-dh-env.sh
```

Test example:
```console
lcdb test --openml-id 61 -w lcdb.workflow.keras.DenseNNWorkflow -m
```

## Experiments

### Example

The Example section provides a walkthrough for running a simple experiment with LCDB 2.0.

First, we generate `100` configurations for the `DenseNNWorkflow`, with the first configuration being the default. The `lcdb create` command handles this, outputting the configurations to `configs.csv`.

```console
lcdb create -w lcdb.workflow.keras.DenseNNWorkflow -n 100 -o configs.csv
```

Next, we run the experiment on the OpenML task with id `3`, using the `DenseNNWorkflow`. We limit the experiment to `100` evaluations and set a time limit of 3600 seconds. The `--initial-configs` argument loads the configurations from `configs.csv` that we previously generated. `--monotonic` tells the search to optimize for monotonically increasing performance over iterations. `--verbose` prints detailed logs during the run.

```console
lcdb run --openml-id 3 -w lcdb.workflow.keras.DenseNNWorkflow --monotonic --max-evals 100 -t 3600 --initial-configs configs.csv --verbose
```

The experiment results are saved to `results.csv`. This file contains the evaluation results for each configuration on the task, including scores, times, and fidelity values. We can load and analyze this file to study how the configurations performed.

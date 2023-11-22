# LCDB 2.0: Learning Curves Database of Configurable Learning Workflows

LCDB 2.0 is a database of learning workflows with diverse configurations (or hyperparameters).

## Installation

The `install/` directory contains the files required for installation.

From this directory

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

## Data Standards

### Results Dataframes

DeepHyper returns `results.csv` files. Such files can be opened with Pandas.

### Representation of Learning Curves

```python
learning_curve = {
    "fidelity_unit": "epochs",
    "fidelity_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score_types": ["loss", "accuracy"],
    "score_values": [...],
    "time_types": ["epoch"],
    "time_values": [...],
}
```

- `fidelity_unit`: a string, describing the unit of the fidelity (e.g., `samples`, `epochs`, `batches`, `resolution`, etc.).
- `fidelity_values`: a 1-D array of reals, giving the fidelity value at which the `score_values` and `time_values` are collected.
- `score_types`: a 1-D array of strings, describing the name(s) of the scoring function(s) collected (e.g., `loss`, `accuracy`, `balanced_accuracy`, etc.).
- `score_values`: a 3-D array of reals, where `axis=0` corresponds to `fidelity_values` and has the same length, where `axis=1` corresponds to data splits `["train", "valid", "test"]` (the 3 are not always present) with a length from 1 to 3, where `axis=2` corresponds to `score_types` and has the same length.
- `time_types`: a 1-D array of strings, where each value describes a type of timing (e.g., `fit`, `predict`, `epoch`).
- `times_values`: a 3-D array of reals, where `axis=0` corresponds to `fidelity_values` and has the same length, where `axis=1` corresponds to data splits `["train", "valid", "test"]` (the 3 are not always present) with a length from 1 to 3, where `axis=2` corresponds to `time_types` and has the same length. **REFINE**
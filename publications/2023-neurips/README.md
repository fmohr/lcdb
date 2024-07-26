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

## Adding Results to your LCDB

Once you got a result file via `lcdb run` with one row per evaluation, say, `results.csv.gz`, you can add these results to your learning curve data base as follows:
```console
lcdb add -c <campaign_name> results.csv.gz
```

The campaign name is only to give a label to the runs (*todo: probably replace or complement by a file that describes the execution conditions*)

By default, the data will be stored in `~/.lcdb/data`. However, you can add a flag `-r <repository_name>` to change the repository to which it will be added. Here `<repository_name>` must be one of the keys of the `repositories` dictionary in your `config.json`, which is looked up by default in `~/.lcdb`. If you have a `.lcdb` folder in the working directory where you place the prompt, then `.lcdb/config.json` is used instead. By default, `<repository_name>` is `home`, which is why `~/.lcdb/data` is used to store results by default.

## Extracting LCDB results in Python
The general logic is to retrieve a dataframe with one row for every evaluation of any hyperparameter configuration contained in the database:
```python
from lcdb.db import LCDB
df = LCDB().get_results()
```
which will fetch all learning curves in the system (there is a protection mechanism that makes sure that no more than 10 million curves will be retrieved).

It is generally recommendable to filter results by workflow, datasets, or both, which can be done by passing those parameters to the `get_results` function:

```python
df = LCDB().get_results(workflows=["lcdb.workflow.sklearn.LibLinearWorkflow"], openmlids=[3, 6])
```

The results dataframe has a field `m:json`, which contains a dictionary with all the information about the curve of the evaluation. if there are iteration curves, they are also contained in that dictionary.

We use [https://jmespath.org/] to conveniently access and retrieve information from this dictionary. Example can be found in `analysis/json.py`.

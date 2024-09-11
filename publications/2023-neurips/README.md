# LCDB 2.0: Learning Curves Database of Configurable Learning Workflows

LCDB 2.0 is a database of learning workflows with diverse configurations (or hyperparameters).

## Installation

Before starting we advise the user to create a Python environment using `conda`.

Then, the `lcdb` Package can be installed withing the environment by running the following command from this directory:
```console
pip install -e "."
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

By default, the data will be stored in `~/.lcdb/data`. However, there are two ways to change this:

1. you can create an alternative LCDB that does not reside in `~/.lcdb`. For this, simply run `lcdb init <path>` with a folder in which you want to use LCDB from the CLI. This may be an existing or unexisting directory name. In either case, the folder `<path>` will not be modified except that a hidden subfolder `.lcdb` is being added, so it will not be flooded with LCDB data, and you can safely apply it to a research working directory of yours.

2. you can add a flag `-r <repository_name>` to change the repository to which it will be added. Here `<repository_name>` must be one of the keys of the `repositories` dictionary in your `config.json`, which is looked up by default in `./lcdb` (in the current working directory) and, if it is not found there, in `~/.lcdb`. So depending on whether you have a local LCDB created with the first option or not, you need to modify `.lcdb/config.json` or `~/.lcdb/config.json` to add the repository.

## Extracting LCDB results in Python

### Retrieving Raw Results
The general logic is to retrieve a dataframe with one row for every evaluation of any hyperparameter configuration contained in the database.
To prevent from memory explosion, these are read in through a generator, once at a time:
```python
from lcdb.db import LCDB
from tqdm import tqdm

for df in tqdm(LCDB().query()):
  # do something with the result dataframe
```
Every result dataframe contains data for exclusively one workflow (but usually not all results available for this workflow). This is because the columns of the dataframe depend on the workflow hyperparameters, and since these are different for different workflows, results are not merged across different workflows by default.

Because of potentially large retrieval times (due to large and compressed amounts of data), results are returned through generators by default.
These generators have a `__len__` attribute so that a progress bar can be used to estimate the retrieval time.

Every result dataframe contains *one column for every hyperparameter of the workflow* (all of these columns begin with an `p:`), and the following workflow-independent columns:

| Column Name    | Description |
| -------- | ------- |
| objective | ?? |
| job_id | ?? |
| m:timestamp_submit | ?? |
| m:timestamp_gather | ?? |
| m:timestamp_start  | ?? | 
| m:timestamp_end    | ?? |
| m:memory           | ?? |
| m:openmlid         | Dataset ID at openml.org for the dataset |
| m:workflow         | Name of the workflow |
| m:workflow_seed    | Seed used to configure the workflow |
| m:valid_prop       | Portion of data separated for validation fold |
| m:test_prop        | Portion of data separated for test fold |
| m:monotonic        | Boolean that indicates whether training folds of higher size are supersets of training sets of lower size (with same validation and test seeds) |
| m:valid_seed       | Seed used to separate training data from validation data (inner split) |
| m:test_seed        | Seed used to separate test data from rest (outer split) |
| m:traceback        | Traceback of the error in case of failure |
| m:json             | Detailed learning curve results as a Python *dictionary* (see below) |

Because of the amount of data available with LCDB, it is generally recommendable to filter results by workflow, datasets, or both, which can be done by passing those parameters to the `query` function:

```python
generator = LCDB().query(workflows=["lcdb.workflow.sklearn.LibLinearWorkflow"], openmlids=[3, 6])
```

Even this way, query times are generally high, and you probably want to avoid many of these queries.
It is therefore highly recommended to retrieve the information you are interested in from the dataframes and only store the important information.
Most applications only need a fraction of the stored information (often less than 10%), so storing the relevant information locally will drastically speed up your research activity.

### Processing Result Dictionaries
The dictionaries stored inside `m:json` contain a potentially deep tree structure. The dictionary itself is the root node of this tree, and for any node, children can be obtained through the `children` key. Each node has the following entries:
| Key    | Type of value  | Description of the value |
| -------- | ------- | ------- |
| timestamp_start | float | The (relative) timestamp indicating at which time the computation for the entries inside the node started. |
| timestamp_stop | float | The (relative) timestamp indicating at which time the computation for the entries inside the node ended. |
| tag (optional) | str | Name tag of the node (not unique, comparable to `class` in CSS) |
| metadata (optional) | dict | Information associated with the node. Content depends on the tag |
| children (optional) | dict | children of the node |

One can use [https://jmespath.org/] to conveniently access and retrieve information from this dictionary. Example can be found in `analysis/json.py`.

The learning curve data is contained in the second child, which has the tag `build_curves`. The general structure of the tree is as follows, where there is one anchor for every $\ceil{2^{\frac{i}{2}}}$ with $i \geq 8$, i.e., starting at 16, up to the maximum size allowed by dataset and `valid_prop` and `test_prop`. The respective values of the anchors can be found in the meta-data of the node. Most nodes are only needed to know the runtimes (everything in `fit` and `get_predictions`). In `metrics`, one can find both the runtimes to compute the metric values as the metric values themselves. They are computed for train, validation, and test data. `confusion_matrix` is the basis for all metrics that are derived from definite predictions such as accuracy, error rate, precision, recall, f1, etc. For space reasons, no probabilistic predictions are memorized, but the values of AUC, log_loss and brier_score are memorized and made explicit. The concrete values are obtained via `["metadata"]["value"]` inside those nodes.

```
├── load_task
└── build_curves
    ├── anchor
    │   ├── create_workflow
    │   ├── fit
    │   │   ├── transform_train
    │   │   ├── transform_valid
    │   │   └── transform_test
    │   ├── get_predictions
    │   │   ├── train
    │   │   │   ├── predict_proba
    │   │   ├── val
    │   │   │   ├── predict_proba
    │   │   └── test
    │   │       ├── predict_proba
    │   └── metrics
    │       ├── train
    │       │   ├── confusion_matrix
    │       │   ├── auc
    │       │   ├── log_loss
    │       │   └── brier_score
    │       ├── val
    │       │   ├── confusion_matrix
    │       │   ├── auc
    │       │   ├── log_loss
    │       │   └── brier_score
    │       └── test
    │           ├── confusion_matrix
    │           ├── auc
    │           ├── log_loss
    │           └── brier_score
    .
    .
    .
    └── anchor
        ├── create_workflow
        ├── fit
        │   ├── transform_train
        │   ├── transform_valid
        │   └── transform_test
        ├── get_predictions
        │   ├── train
        │   │   ├── predict_proba
        │   ├── val
        │   │   ├── predict_proba
        │   └── test
        │       ├── predict_proba
        └── metrics
            ├── train
            │   ├── confusion_matrix
            │   ├── auc
            │   ├── log_loss
            │   └── brier_score
            ├── val
            │   ├── confusion_matrix
            │   ├── auc
            │   ├── log_loss
            │   └── brier_score
            └── test
                ├── confusion_matrix
                ├── auc
                ├── log_loss
                └── brier_score
```
### Using processors for computations prior to result delivery
When calling the `query` function, you can add a dictionary of callables that will be applied to each dictionary inside of `m:json`. To do so, use the `processors` attribute.
Importantly, when using `processors`, the column `m:json` is *removed* before returning the results, which can save memory, specificall if you do not want to use a generator (see below).

The following code generates two new columns with names `anchor_sizes` and `balanced_error_rate` in the result dataframe:

```python
df_gen = lcdb.query(processors={
  "anchor_sizes": QueryAnchorValues(),
  "balanced_error_rate": lambda x: list(map(lambda x: 1 - balanced_accuracy_from_confusion_matrix(x), QueryMetricValuesFromAnchors("confusion_matrix", split_name="val")(x)))
})
```



### Retrieving results without a generator
It is highly recommended to use the generators for data retrieval. However, if you prefer to load the data in a batch fashion, you can proceed as follows:

```python
df = lcdb.query(workflows="<name of one workflow>", openmlids=[...], return_generator=False)
```
if you want data for just one workflow or

```python
df_dict = lcdb.query(workflows=["<wf 1>", "<wf 2>"], openmlids=[...], return_generator=False)
```
if it is for several workflows. In the second case, you get a dictionary in which the keys are the workflow names and the values are the dataframes with all results for those workflows.

When not using a generator, it is highly recommended to use *processors* (described above) to make sure that the `m:json` column is discarded and hence to avoid memory flooding.

### Retrieving Learning Curves in Numpy Format
It is also possible to obtain learning curves already in a more aggregated way through objects of the `lcdb.db.util.LearningCurve` class.
Objects of these class are basically annotated numpy arrays.
If `lc` is an object of that class, then `lc.values` is a numpy array with 6 or 7 dimensions with the following meanings:
1. the metrics for which values are available (names accessible via `lc.metrics`)
2. the folds for which values are available (names accessible via `lc.fold_names`)
3. the seeds for test-data separation for which values are available (seed values accessible via `lc.test_seeds`)
4. the seeds for train/validation split for which values are available (seed values accessible via `lc.val_seeds`)
5. the seeds for workflow random states for which values are available (seed values accessible via `lc.workflow_seeds`)
6. the sample-wise anchors for the (absolute) training set sizes for which values are available (anchors accessible via `lc.anchors_size`), and
7. the iteration-wise anchors (epochs/number of trees, etc.) for which values are available (anchors accessible via `lc.anchors_iteration`), only for workflows that produce iteration-wise learning curves.

The availablility of iteration-wise anchors can be asked through the boolean `lc.is_iteration_wise_curve`, which is True iff there is a 7-th dimension for the iteration anchors.

A learning curve object can only store information for *a single hyperparameter configuration of a single workflow on a single dataset*. These three attributes can also be accessed via:
- `lc.workflow`
- `lc.hp_config`
- `lc.openmlid`

There is an extractor that will automatically create objects of this type from the result rows:
```python
from lcdb.analysis.util import LearningCurveExtractor
lcdb = LCDB()
lcdb.query(
    workflows=workflow,
    processors={
        "learning_curve": LearningCurveExtractor(
            metrics=["error_rate"],
            folds=["train", "val", "test", "oob"]
        )
    }
)
```

Since results are delivered for all different seed combinations, it can be convenient to aggregate the results for a single dataset/hp-configuration combination into a single LearningCurve object.
A utility function `lcdb.db.util.merge_curves` allows to do this, and this can also be used directly with standard Pandas functions:

```python
from lcdb.analysis.util import merge_curves

config_cols = [c for c in df.columns if c.startswith("p:")]
df.groupby(config_cols).agg({"learning_curve": merge_curves})
```
After this operation, there will be only one line for every hyperparameter configuration in the dataframe, and learning curves for different seeds but otherwise identical setups will have been merged into a single learning curve object (missing values will be nan).

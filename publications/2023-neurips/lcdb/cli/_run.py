"""Command line to run experiments."""
import logging
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from lcdb.data import load_task
from lcdb.data.split import train_valid_test_split
from lcdb.utils import import_attr_from_module
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "run"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Run experiments with DeepHyper."
    )

    subparser.add_argument("-id", "--openml-id", type=int, required=True)
    subparser.add_argument("-w", "--workflow-class", type=str, required=True)
    subparser.add_argument(
        "-m", "--monotonic", action="store_true", default=False, required=False
    )
    subparser.add_argument("-vs", "--valid-seed", type=int, default=42, required=False)
    subparser.add_argument("-ts", "--test-seed", type=int, default=42, required=False)
    subparser.add_argument(
        "-vp", "--valid-prop", type=float, default=0.1, required=False
    )
    subparser.add_argument(
        "-tp", "--test-prop", type=float, default=0.1, required=False
    )
    subparser.add_argument("-d", "--log-dir", type=str, default=".", required=False)
    subparser.add_argument(
        "--max-evals",
        type=int,
        default=100,
        required=False,
        help="Number of configurations to run",
    )
    subparser.add_argument("-t", "--timeout", type=int, default=1800, required=False)
    subparser.add_argument("--initial-configs", type=str, required=False, default=None)
    subparser.add_argument(
        "-v", "--verbose", action="store_true", default=False, required=False
    )

    subparser.set_defaults(func=function_to_call)


def get_anchor_schedule(n):
    """Get a schedule of anchors for a given size `n`."""
    anchors = []
    k = 1
    while True:
        exponent = (7 + k) / 2
        sample_size = int(np.round(2**exponent))
        if sample_size > n:
            break
        anchors.append(sample_size)
        k += 1
    if anchors[-1] < n:
        anchors.append(n)
    return anchors


@profile(memory=True)
def run(
    job: RunningJob,
    openml_id: int = 3,
    workflow_class: str = "lcdb.workflow.sklearn.LibLinearWorkflow",
    monotonic: bool = True,
    valid_seed: int = 42,
    test_seed: int = 42,
    valid_prop: float = 0.1,
    test_prop: float = 0.1,
):
    # Load the workflow
    WorkflowClass = import_attr_from_module(workflow_class)

    # Load the raw dataset
    (X, y), dataset_metadata = load_task(f"openml.{openml_id}")
    num_instances = X.shape[0]

    # Transform categorical features
    categories = np.asarray(dataset_metadata["categories"], dtype=bool)

    if not (np.any(categories)):
        one_hot_encoder = FunctionTransformer(func=lambda x: x, validate=False)
    else:
        one_hot_encoder = OneHotEncoder(
            drop="first", sparse_output=False
        )  # TODO: drop "first" could be an hyperparameter
        one_hot_encoder.fit(X[:, categories])

    anchors = get_anchor_schedule(int(num_instances * (1 - test_prop - valid_prop)))

    # Infos to be collected
    # Trying to normalize the format
    infos = {
        "fidelity_unit": "samples",
        "fidelity_values": [],
        "score_types": ["accuracy", "loss"],
        "score_values": [],
        "time_types": ["fit", "predict"],
        "time_values": [],
    }

    # Build sample-wise learning curve
    for i, anchor in enumerate(anchors):
        # Split the dataset
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
            X, y, test_seed, valid_seed, test_prop, valid_prop, stratify=True
        )

        logging.info(
            f"Running anchor {anchor} which is {anchor/X_train.shape[0]*100:.2f}% of the dataset."
        )

        train_idx = np.arange(X_train.shape[0])
        # If not monotonic, the training set should be shuffled differently for each anchor
        # so that the training sets of different anchors do not contain eachother
        if not (monotonic):
            random_seed_train_shuffle = np.random.RandomState(valid_seed).randint(
                0, 2**32 - 1, size=len(anchors)
            )[i]
            rs = np.random.RandomState(random_seed_train_shuffle)
            rs.shuffle(train_idx)

            X_train = X_train[train_idx]
            y_train = y_train[train_idx]

        X_train = X_train[:anchor]
        y_train = y_train[:anchor]

        # Preprocessing
        # TODO: could be a module of a workflow
        logging.info("Preprocessing the data...")
        # 1. Start by fitting the scaler on the training set
        X_train_cat = one_hot_encoder.transform(X_train[:, categories])
        X_train = np.concatenate([X_train_cat, X_train[:, ~categories]], axis=1)

        # 2. Then transform the validation and test sets
        X_valid_cat = one_hot_encoder.transform(X_valid[:, categories])
        X_valid = np.concatenate([X_valid_cat, X_valid[:, ~categories]], axis=1)

        X_test_cat = one_hot_encoder.transform(X_test[:, categories])
        X_test = np.concatenate([X_test_cat, X_test[:, ~categories]], axis=1)

        # Create and fit the workflow
        logging.info("Creating and fitting the workflow...")
        workflow = WorkflowClass(**job.parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow.fit(X_train, y_train)

        time_fit = workflow.infos["fit_time"]

        # Predict and Score
        logging.info("Predicting and scoring...")
        scores = []
        for i, (X_, y_true) in enumerate(
            [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]
        ):
            y_pred = workflow.predict(X_)

            if i == 0:
                time_predict = workflow.infos["predict_time"]

            accuracy = round(accuracy_score(y_true, y_pred), ndigits=5)
            loss = round(zero_one_loss(y_true, y_pred), ndigits=5)
            scores.append([accuracy, loss])

        # Collect Infos
        infos["fidelity_values"].append(anchor)
        infos["score_values"].append(scores)
        infos["time_values"].append([time_fit, time_predict])

    # Other infos
    infos["dataset_id"] = openml_id
    infos["workflow"] = workflow_class
    infos["valid_prop"] = valid_prop
    infos["test_prop"] = valid_prop
    infos["monotonic"] = monotonic
    infos["valid_seed"] = valid_seed
    infos["test_seed"] = test_seed

    # Meaning of indexes
    # -1: last fidelity, 1: validation set, 0: accuracy
    valid_accuracy = infos["score_values"][-1][1][0]
    results = {"objective": valid_accuracy, "metadata": infos}

    return results


def main(
    openml_id,
    workflow_class,
    monotonic,
    valid_seed,
    test_seed,
    valid_prop,
    test_prop,
    log_dir,
    max_evals,
    timeout,
    initial_configs,
    verbose,
):
    """
    :meta private:
    """
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, "deephyper.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )

    # Load the workflow to get its config space
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()
    config_default = config_space.get_default_configuration().get_dictionary()

    # Set the search space
    problem = HpProblem(config_space)

    # Initial Configs
    initial_points = []
    if os.path.exists(initial_configs):
        ip_df = pd.read_csv(initial_configs)
        ip_df = ip_df[problem.hyperparameter_names]
        for _, row in ip_df.iterrows():
            initial_points.append(row.to_dict())
    else:
        initial_points.append(config_default.get_dictionary()) 

    evaluator = Evaluator.create(
        run,
        method="ray",
        method_kwargs={
            "address": "auto",
            # "num_cpus": 10,
            "num_cpus_per_task": 1,
            "run_function_kwargs": {
                "openml_id": openml_id,
                "workflow_class": workflow_class,
                "monotonic": monotonic,
                "valid_seed": valid_seed,
                "test_seed": test_seed,
                "valid_prop": valid_prop,
                "test_prop": test_prop,
            },
            "callbacks": [TqdmCallback()] if verbose else [],
        },
    )

    # Set the search algorithm
    search = CBO(
        problem,
        evaluator,
        log_dir=log_dir,
        initial_points=initial_points,
        surrogate_model="DUMMY",
        verbose=verbose,
    )

    # Execute the search
    results = search.search(max_evals, timeout=timeout)


def test_default_config_run_deephyper():
    workflow_class = "lcdb.workflow.sklearn.LibLinearWorkflow"
    workflow_class = import_attr_from_module(workflow_class)
    config_space = workflow_class.get_config_space()
    config_default = config_space.get_default_configuration().get_dictionary()

    # id 3, 6 are good tests
    output = run(RunningJob(id=0, parameters=config_default), openml_id=3, verbose=0)
    import pprint

    pprint.pprint(output)


def test_random_sampling():
    from deephyper.evaluator import Evaluator
    from deephyper.evaluator.callback import TqdmCallback
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    # Experiment Parameters
    openml_id = 3
    workflow_class = "lcdb.workflow.sklearn.LibLinearWorkflow"

    # Load the workflow to get its config space
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()
    config_default = config_space.get_default_configuration().get_dictionary()

    # Set the search space
    problem = HpProblem(config_space)

    evaluator = Evaluator.create(
        run,
        method="ray",
        method_kwargs={
            "address": "auto",
            # "num_cpus": 10,
            "num_cpus_per_task": 1,
            "run_function_kwargs": {
                "openml_id": openml_id,
                "workflow_class": workflow_class,
            },
            "callbacks": [TqdmCallback()],
        },
    )

    # Set the search algorithm
    search = CBO(
        problem,
        evaluator,
        log_dir="test-dh",
        initial_points=[config_default],
        surrogate_model="DUMMY",
        verbose=1,
    )

    # Execute the search
    results = search.search(100)


if __name__ == "__main__":
    # logging.basicConfig(
    #     # filename=path_log_file, # optional if we want to store the logs to disk
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    #     force=True,
    # )
    test_random_sampling()

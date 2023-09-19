"""Command line to run experiments."""
import functools
import json
import logging
import os
import pathlib
import traceback
import warnings

import numpy as np
import pandas as pd
from deephyper.core.utils._timeout import terminate_on_timeout
from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from lcdb.data import load_task
from lcdb.data.split import train_valid_test_split
from lcdb.utils import import_attr_from_module
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Avoid Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "run"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Run experiments with DeepHyper."
    )

    subparser.add_argument(
        "-id",
        "--openml-id",
        type=int,
        required=True,
        help="The identifier of the OpenML dataset.",
    )
    subparser.add_argument(
        "-w",
        "--workflow-class",
        type=str,
        required=True,
        help="The 'path' of the workflow to train.",
    )
    subparser.add_argument(
        "-m",
        "--monotonic",
        action="store_true",
        default=False,
        required=False,
        help="A boolean indicating if the sample-wise learning curve should be monotonic (i.e., sample set at smaller anchors are always included in sample sets at larger anchors) or not.",
    )
    subparser.add_argument(
        "-vs",
        "--valid-seed",
        type=int,
        default=42,
        required=False,
        help="Random state seed of train/validation split.",
    )
    subparser.add_argument(
        "-ts",
        "--test-seed",
        type=int,
        default=42,
        required=False,
        help="Random state seed of train+validation/test split.",
    )
    subparser.add_argument(
        "-ws",
        "--workflow-seed",
        type=int,
        default=42,
        required=False,
        help="Random state seed of the workflow.",
    )
    subparser.add_argument(
        "-vp",
        "--valid-prop",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of validation/(train+validation).",
    )
    subparser.add_argument(
        "-tp",
        "--test-prop",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of test/data.",
    )
    subparser.add_argument(
        "--timeout-on-fit",
        type=int,
        default=-1,
        required=False,
        help="Timeout in seconds for the fit method. Defaults to -1 for unlimited time.",
    )
    subparser.add_argument(
        "-d",
        "--log-dir",
        type=str,
        default=".",
        required=False,
        help="Directory where to store the outputs/logs.",
    )
    subparser.add_argument(
        "--max-evals",
        type=int,
        default=100,
        required=False,
        help="Number of configurations to run.",
    )
    subparser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=1800,
        required=False,
        help="Overall timeout in seconds.",
    )
    subparser.add_argument(
        "--initial-configs",
        type=str,
        required=False,
        default=None,
        help="Path to a CSV file containing initial configurations.",
    )
    subparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        required=False,
        help="Boolean to activate or not the verbose mode.",
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
    workflow_seed: int = 42,
    valid_prop: float = 0.1,
    test_prop: float = 0.1,
    timeout_on_fit=-1,
    known_categories: bool = True,
    raise_errors: bool = False,
):
    """This function trains the workflow on a dataset and returns performance metrics.

    Args:
        job (RunningJob): A running job passed by DeepHyper (represent an instance of the function).
        openml_id (int, optional): The identifier of the OpenML dataset. Defaults to 3.
        workflow_class (str, optional): The "path" of the workflow to train. Defaults to "lcdb.workflow.sklearn.LibLinearWorkflow".
        monotonic (bool, optional): A boolean indicating if the sample-wise learning curve should be monotonic (i.e., sample set at smaller anchors are always included in sample sets at larger anchors) or not. Defaults to True.
        valid_seed (int, optional): Random state seed of train/validation split. Defaults to 42.
        test_seed (int, optional): Random state seed of train+validation/test split. Defaults to 42.
        workflow_seed (int, optional): Random state seed of the workflow. Defaults to 42.
        valid_prop (float, optional): Ratio of validation/(train+validation). Defaults to 0.1.
        test_prop (float, optional): Ratio of test/data . Defaults to 0.1.
        timeout_on_fit (int, optional): Timeout in seconds for the fit method. Defaults to -1 for infinite time.
        known_categories (bool, optional): If all the possible categories are assumed to be known in advance. Defaults to True.
        raise_errors (bool, optional): If `True`, then errors are risen to the outside. Otherwise, just a log message is generated. Defaults to False.

    Returns:
        dict: a dictionnary with 2 keys (objective, metadata) where objective is the objective maximized by deephyper (if used) and metadata is a JSON serializable sub-dictionnary which are complementary information about the workflow.
    """
    # Load the workflow
    WorkflowClass = import_attr_from_module(workflow_class)

    # Load the raw dataset
    (X, y), dataset_metadata = load_task(f"openml.{openml_id}")
    num_instances = X.shape[0]

    # Transform categorical features
    columns_categories = np.asarray(dataset_metadata["categories"], dtype=bool)
    values_categories = None

    dataset_metadata["categories"] = {"columns": columns_categories}
    if not (np.any(columns_categories)):
        one_hot_encoder = FunctionTransformer(func=lambda x: x, validate=False)
    else:
        dataset_metadata["categories"]["values"] = None
        one_hot_encoder = OneHotEncoder(
            drop="first", sparse_output=False
        )  # TODO: drop "first" could be an hyperparameter
        one_hot_encoder.fit(X[:, columns_categories])
        if known_categories:
            values_categories = one_hot_encoder.categories_
            values_categories = [v.tolist() for v in values_categories]
            dataset_metadata["categories"]["values"] = values_categories

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
        "child_fidelities": [],
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
        if not monotonic:
            random_seed_train_shuffle = np.random.RandomState(valid_seed).randint(
                0, 2**32 - 1, size=len(anchors)
            )[i]
            rs = np.random.RandomState(random_seed_train_shuffle)
            rs.shuffle(train_idx)

            X_train = X_train[train_idx]
            y_train = y_train[train_idx]

        X_train = X_train[:anchor]
        y_train = y_train[:anchor]

        # Create and fit the workflow
        logging.info("Creating and fitting the workflow...")
        workflow_kwargs = job.parameters
        workflow_kwargs["random_state"] = workflow_seed
        workflow = WorkflowClass(**workflow_kwargs)

        if timeout_on_fit > 0:
            workflow.fit = functools.partial(
                terminate_on_timeout, timeout_on_fit, workflow.fit
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                if workflow.requires_valid_to_fit:
                    workflow.fit(
                        X_train,
                        y_train,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        metadata=dataset_metadata,
                    )
                else:
                    workflow.fit(X_train, y_train, metadata=dataset_metadata)
            except Exception as e:
                if raise_errors:
                    raise

                # Collect child fidelities (e.g., iteration learning curves with epochs) if available
                if hasattr(workflow, "fidelities"):
                    infos["child_fidelities"].append(workflow.fidelities)

                infos["dataset_id"] = openml_id
                infos["workflow"] = workflow_class
                infos["valid_prop"] = valid_prop
                infos["test_prop"] = valid_prop
                infos["monotonic"] = monotonic
                infos["valid_seed"] = valid_seed
                infos["test_seed"] = test_seed
                infos["workflow_seed"] = workflow_seed
                infos["traceback"] = traceback.format_exc()

                logging.error(
                    f"Error while fitting the workflow: \n{infos['traceback']}"
                )

                infos["traceback"] = r"{}".format(infos["traceback"])

                if (
                    len(infos["score_values"]) > 0
                    and len(infos["score_values"][-1]) > 0
                ):
                    # -1: last fidelity, 1: validation set, 0: accuracy
                    objective = infos["score_values"][-1][1][0]
                else:
                    objective = "F"

                return {"objective": objective, "metadata": infos}

        time_fit = workflow.infos["fit_time"]

        # Predict and Score
        logging.info("Predicting and scoring...")
        scores = []
        for i, (X_, y_true) in enumerate(
            [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

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

        # Collect child fidelities (e.g., iteration learning curves with epochs) if available
        if hasattr(workflow, "fidelities"):
            infos["child_fidelities"].append(workflow.fidelities)

    # Other infos
    infos["dataset_id"] = openml_id
    infos["workflow"] = workflow_class
    infos["valid_prop"] = valid_prop
    infos["test_prop"] = valid_prop
    infos["monotonic"] = monotonic
    infos["valid_seed"] = valid_seed
    infos["test_seed"] = test_seed
    infos["workflow_seed"] = workflow_seed
    infos["traceback"] = ""

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
    workflow_seed,
    valid_prop,
    test_prop,
    timeout_on_fit,
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
    if initial_configs is not None and os.path.exists(initial_configs):
        ip_df = pd.read_csv(initial_configs)
        ip_df = ip_df[problem.hyperparameter_names]
        for _, row in ip_df.iterrows():
            initial_points.append(row.to_dict())
    else:
        initial_points.append(config_default)

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
                "workflow_seed": workflow_seed,
                "valid_prop": valid_prop,
                "test_prop": test_prop,
                "timeout_on_fit": timeout_on_fit,
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


def test_default_config():
    workflow_class = "lcdb.workflow.sklearn.LibLinearWorkflow"
    # workflow_class = "lcdb.workflow.keras.DenseNNWorkflow"
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()
    config_default = config_space.get_default_configuration().get_dictionary()

    # config_default["transform_cat"] = "ordinal"
    # config_default["optimizer"] = "Ftrl"
    config_default = {
        "p:C": 6.165666572362732,
        "p:class_weight": "balanced",
        "p:dual": False,
        "p:fit_intercept": False,
        "p:intercept_scaling": 33.513181992610626,
        "p:loss": "squared_hinge",
        "p:max_iter": 4536,
        "p:multiclass": "ovo-scikit",
        "p:penalty": "l2",
        #! ERROR
        # "p:pp_cat_encoder": "ordinal",
        # "p:pp_decomposition": "lda",
        # "p:pp_featuregen": "none",
        # "p:pp_featureselector": "select50",
        # "p:pp_scaler": "std",
        #!
        "p:pp_cat_encoder": "ordinal",
        "p:pp_decomposition": "none",
        "p:pp_featuregen": "poly2",
        "p:pp_featureselector": "generic",
        "p:pp_scaler": "minmax",
        "p:tol": 0.0013005105948736,
        "objective": "F",
        "job_id": 6,
        "m:timestamp_submit": 1.7238471508026123,
        "m:timestamp_gather": 6.4399919509887695,
        "m:timestamp_start": 1694424265.106677,
        "m:timestamp_end": 1694424266.134169,
        "m:memory": 10653821,
    }
    config_default = {k[2:]: v for k, v in config_default.items() if k.startswith("p:")}

    # id 3, 6 are good tests
    output = run(
        RunningJob(id=0, parameters=config_default),
        openml_id=3,
        workflow_class=workflow_class,
        raise_errors=True,
    )
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
    test_default_config()

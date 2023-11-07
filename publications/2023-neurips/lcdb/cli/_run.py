"""Command line to run experiments."""
import copy
import logging
import os
import pathlib

import json
import pandas as pd
from deephyper.core.exceptions import SearchTerminationError
from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from lcdb.data import load_task
from lcdb.LCController import LCController
from lcdb.utils import import_attr_from_module

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
    subparser.add_argument(
        "-e",
        "--evaluator",
        default="ray",
        type=str,
        help="The evaluator to use. It can be 'serial', 'thread', 'process', 'ray' or 'mpicomm'.",
    )
    subparser.add_argument(
        "--num-workers",
        default=-1,
        type=int,
        help="The number of workers to use with the evaluator. Defaults to -1 for all available workers.",
    )
    subparser.set_defaults(func=function_to_call)


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
        dict: a dictionary with 2 keys (objective, metadata) where objective is the objective maximized by deephyper (if used) and metadata is a JSON serializable sub-dictionnary which are complementary information about the workflow.
    """

    infos = {"openmlid": openml_id, "workflow_seed": workflow_seed}

    # Load the raw dataset
    (X, y), dataset_metadata = load_task(f"openml.{openml_id}")

    # Create and fit the workflow
    logging.info("Creating the workflow...")
    WorkflowClass = import_attr_from_module(workflow_class)
    workflow_kwargs = copy.deepcopy(job.parameters)
    workflow_kwargs["random_state"] = workflow_seed
    workflow = WorkflowClass(**workflow_kwargs)

    # create controller
    controller = LCController(
        workflow=workflow,
        X=X,
        y=y,
        dataset_metadata=dataset_metadata,
        test_seed=test_seed,
        valid_seed=valid_seed,
        monotonic=monotonic,
        test_prop=test_prop,
        valid_prop=valid_prop,
        timeout_on_fit=timeout_on_fit,
        known_categories=known_categories,
        stratify=True,
        raise_errors=raise_errors,
    )

    # build the curves
    controller.build_curves()

    # update infos based on report
    infos.update(controller.report)

    # TODO: to be replaced by the real score(s)
    # get validation accuracy score on last anchor
    valid_accuracy = controller.objective
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
    evaluator,
    num_workers,
):
    """Entry point for the command line interface."""

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

    run_function_kwargs = {
        "openml_id": openml_id,
        "workflow_class": workflow_class,
        "monotonic": monotonic,
        "valid_seed": valid_seed,
        "test_seed": test_seed,
        "workflow_seed": workflow_seed,
        "valid_prop": valid_prop,
        "test_prop": test_prop,
        "timeout_on_fit": timeout_on_fit,
    }

    if evaluator in ["thread", "process", "ray"]:
        if num_workers < 0:
            if hasattr(os, "sched_getaffinity"):
                # Number of CPUs the current process can use
                num_workers = len(os.sched_getaffinity(0))
            else:
                num_workers = os.cpu_count()

        if evaluator == "ray":
            method_kwargs = {
                "address": os.environ.get("RAY_ADDRESS", None),
                "num_cpus": num_workers,
                "num_cpus_per_task": 1,
            }
        else:
            method_kwargs = {"num_workers": num_workers}
    else:  # mpicomm
        method_kwargs = {}
        if num_workers > 0:
            method_kwargs["num_workers"] = num_workers

    method_kwargs["run_function_kwargs"] = run_function_kwargs
    method_kwargs["callbacks"] = [TqdmCallback()] if verbose else []

    with Evaluator.create(
        run,
        method=evaluator,
        method_kwargs=method_kwargs,
    ) as evaluator:
        # Required for MPI just the root rank will execute the search
        # other ranks will be considered as workers
        if evaluator is not None:
            # Set the search algorithm

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
            results = search.search(max_evals, timeout=timeout, max_evals_strict=True)


def test_default_config():
    # workflow_class = "lcdb.workflow.xgboost.XGBoostWorkflow"
    # workflow_class = "lcdb.workflow.sklearn.KNNWorkflow"
    #    workflow_class = "lcdb.workflow.sklearn.LibLinearWorkflow"
    workflow_class = "lcdb.workflow.sklearn.LibSVMWorkflow"
    # workflow_class = "lcdb.workflow.keras.DenseNNWorkflow"
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()
    config_default = config_space.get_default_configuration().get_dictionary()

    # id 3, 6 are good tests
    output = run(
        RunningJob(id=0, parameters=config_default),
        openml_id=3,
        workflow_class=workflow_class,
        raise_errors=True,
    )

    # check that the output can indeed be compiled into a string using JSON
    json.dumps(output)

    import pprint

    pprint.pprint(output)


if __name__ == "__main__":
    # logging.basicConfig(
    #     # filename=path_log_file, # optional if we want to store the logs to disk
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    #     force=True,
    # )
    test_default_config()

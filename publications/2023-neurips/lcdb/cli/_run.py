"""Command line to run experiments."""

import copy
import logging
import os
import pathlib
import numpy as np

try:
    # Avoid some errors on some MPI implementations
    import mpi4py

    mpi4py.rc.initialize = False
    mpi4py.rc.threads = True
    mpi4py.rc.thread_level = "multiple"
    mpi4py.rc.recv_mprobe = False
    MPI4PY_IMPORTED = True
except ModuleNotFoundError:
    MPI4PY_IMPORTED = False

import lcdb.json
import pandas as pd
from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from lcdb.controller import LCController
from lcdb.data import load_task
from lcdb.timer import Timer
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
        "-tt",
        "--task-type",
        type=str,
        required=False,
        choices=["classification", "regression"],
        default="classification",
        help="The type of the supervised ML task. Either 'classification' or 'regression'.",
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
        help="Ratio of validation/(train+validation+test).",
    )
    subparser.add_argument(
        "-tp",
        "--test-prop",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of test/(train+validation+test).",
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
    subparser.add_argument(
        "--anchor-schedule",
        default="power",
        type=str,
        help="The type of schedule for anchors (over samples of the dataset). Value in ['linear', 'last', 'power'].",
    )
    subparser.add_argument(
        "--epoch-schedule",
        default="power",
        type=str,
        help="The type of schedule for anchors (over learning iterations of the workflow). Value in ['linear', 'last', 'power'].",
    )
    subparser.set_defaults(func=function_to_call)


def run(
    job: RunningJob,
    openml_id: int = 3,
    task_type: str = "classification",
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
    anchor_schedule: str = "power",
    epoch_schedule: str = "power",
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
        anchor_schedule (str, optional): A type of schedule for anchors (over samples of the dataset). Defaults to "power".
        epoch_schedule (str, optional): A type of schedule for epochs (over epochs of the dataset). Defaults to "power".

    Returns:
        dict: a dictionary with 2 keys (objective, metadata) where objective is the objective maximized by deephyper (if used) and metadata is a JSON serializable sub-dictionnary which are complementary information about the workflow.
    """
    logging.info(f"Running job {job.id} with parameters: {job.parameters}")

    timer = Timer(precision=4)
    run_timer_id = timer.start("run")

    # Load the raw dataset
    with timer.time("load_task"):
        logging.info("Loading the dataset...")
        (X, y), dataset_metadata = load_task(f"openml.{openml_id}")

    # Create and fit the workflow
    logging.info("Importing the workflow...")
    WorkflowClass = import_attr_from_module(workflow_class)
    workflow_kwargs = copy.deepcopy(job.parameters)
    workflow_kwargs["epoch_schedule"] = epoch_schedule
    workflow_kwargs["random_state"] = workflow_seed
    workflow_factory = lambda: WorkflowClass(timer=timer, **workflow_kwargs)

    # Initialize information to be returned
    infos = {
        "openmlid": openml_id,
        "workflow_seed": workflow_seed,
        "workflow": workflow_class,
    }

    # create controller
    if task_type not in ["classification", "regression"]:
        raise ValueError(
            f"Task type must be 'classification' or 'regression' but is {task_type}."
        )
    is_classification = task_type == "classification"
    stratify = is_classification

    controller = LCController(
        timer=timer,
        workflow_factory=workflow_factory,
        is_classification=is_classification,
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
        stratify=stratify,
        raise_errors=raise_errors,
        anchor_schedule=anchor_schedule,
    )

    # build the curves
    controller.build_curves()

    assert (
        timer.active_node.id == run_timer_id
    ), f"Timer is not at the right place: {timer.active_node}"
    timer.stop()

    # update infos based on report
    infos.update(controller.report)

    infos["json"] = timer.as_json()

    results = {"objective": controller.objective, "metadata": infos}

    return results


def main(
    openml_id,
    task_type,
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
    anchor_schedule,
    epoch_schedule,
):
    """Entry point for the command line interface."""

    if evaluator in ["serial", "thread", "process", "ray"]:
        # Master-Worker Parallelism: only 1 process will run this code
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, "deephyper.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

        if num_workers < 0:
            if evaluator == "serial":
                num_workers = 1
            elif hasattr(os, "sched_getaffinity"):
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
    elif evaluator == "mpicomm":
        # MPI Parallelism: all processes will run this code
        method_kwargs = {}
        if num_workers > 0:
            method_kwargs["num_workers"] = num_workers

        from mpi4py import MPI

        if not MPI.Is_initialized():
            MPI.Init_thread()

        if MPI.COMM_WORLD.Get_rank() == 0:
            # Only the root rank will create the directory
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        MPI.COMM_WORLD.barrier()  # Synchronize all processes
        logging.basicConfig(
            filename=os.path.join(
                log_dir, f"deephyper.{MPI.COMM_WORLD.Get_rank()}.log"
            ),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")

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
        # TODO: this will fail, needs special preprocessing for inactive parameters such as in lcdb test
        initial_points.append(config_default)

    run_function_kwargs = {
        "openml_id": openml_id,
        "task_type": task_type,
        "workflow_class": workflow_class,
        "monotonic": monotonic,
        "valid_seed": valid_seed,
        "test_seed": test_seed,
        "workflow_seed": workflow_seed,
        "valid_prop": valid_prop,
        "test_prop": test_prop,
        "timeout_on_fit": timeout_on_fit,
        "anchor_schedule": anchor_schedule,
        "epoch_schedule": epoch_schedule,
    }

    method_kwargs["run_function_kwargs"] = run_function_kwargs
    method_kwargs["callbacks"] = [TqdmCallback()] if verbose else []

    # Imposing memory limits to the run-function
    # TODO: memory_limit should replaced and passed as a parameter
    run_function = profile(
        memory=True, memory_limit=0.7 * (1024**3), memory_tracing_interval=0.1
    )(run)

    with Evaluator.create(
        run_function,
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
    # workflow_class = "lcdb.workflow.sklearn.LibLinearWorkflow"
    # workflow_class = "lcdb.workflow.sklearn.LibSVMWorkflow"
    workflow_class = "lcdb.workflow.keras.DenseNNWorkflow"
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()
    config_default = dict(config_space.get_default_configuration())

    # id 3, 6 are good tests
    output = run(
        RunningJob(id=0, parameters=config_default),
        openml_id=61,
        workflow_class=workflow_class,
        raise_errors=True,
    )

    # check that the output can indeed be compiled into a string using JSON
    out = lcdb.json.dumps(output, indent=2)
    print(out)

    lcdb.json.loads(out)


if __name__ == "__main__":
    # logging.basicConfig(
    #     # filename=path_log_file, # optional if we want to store the logs to disk
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    #     force=True,
    # )
    test_default_config()

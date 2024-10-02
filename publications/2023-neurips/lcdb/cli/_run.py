"""Command line to run experiments."""

import logging
import os

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
        "--workflow-memory-limit",
        type=float,
        default=1750.0,
        required=False,
        help="Memory limit per config (MBs).",
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

    subparser.add_argument(
        "--no-exception-on-unsuitable-preprocessor",
        action="store_true",
        default=False,
        required=False,
        help="If set, no exception will be generated if a pre-processor that is irrelevant or useless for the data is being used, e.g., a categorical encoder for a dataset with numerical features only."
    )

    subparser.set_defaults(func=function_to_call)


def run_learning_workflow_from_deephyper(
        job,
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
        raise_exception_on_unsuitable_preprocessor: bool = True,
        anchor_schedule: str = "power",
        epoch_schedule: str = "power",
        logger=None,
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
    if logger is None:
        logger = logging.getLogger("LCDB")
    logger.info(f"Running job {job.id} with parameters: {job.parameters}")

    from lcdb.builder import run_learning_workflow

    return run_learning_workflow(
        openml_id=openml_id,
        task_type=task_type,
        workflow_class=workflow_class,
        workflow_parameters=job.parameters,
        monotonic=monotonic,
        valid_seed=valid_seed,
        test_seed=test_seed,
        workflow_seed=workflow_seed,
        valid_prop=valid_prop,
        test_prop=test_prop,
        timeout_on_fit=timeout_on_fit,
        known_categories=known_categories,
        raise_errors=raise_errors,
        raise_exception_on_unsuitable_preprocessor=raise_exception_on_unsuitable_preprocessor,
        anchor_schedule=anchor_schedule,
        epoch_schedule=epoch_schedule
    )


def run_experiment(
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
    logger,
    evaluator,
    num_workers,
    anchor_schedule,
    epoch_schedule,
    workflow_memory_limit,
    no_exception_on_unsuitable_preprocessor
):

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

    import functools
    import pathlib

    import pandas as pd

    from deephyper.evaluator import Evaluator
    from deephyper.evaluator.callback import TqdmCallback
    from deephyper.hpo import CBO, HpProblem
    from deephyper.hpo._problem import convert_to_skopt_space

    from lcdb.builder.utils import import_attr_from_module, terminate_on_memory_exceeded

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
    if initial_configs is not None:
        if not os.path.exists(initial_configs):
            raise ValueError(
                f"Specified file for initial configs {initial_configs} does not exist!"
            )
        ip_df = pd.read_csv(initial_configs)
        ip_df = ip_df[problem.hyperparameter_names]
        for _, row in ip_df.iterrows():
            initial_points.append(row.to_dict())
    else:
        # Add the default configuration
        # Convert the config space to a skopt space
        skopt_space = convert_to_skopt_space(config_space, surrogate_model="RF")

        config_default = problem.default_configuration
        for i, k in enumerate(skopt_space.dimension_names):
            # Check if hyperparameter k is active
            # If it is not active we attribute the "lower bound value" of the space
            # To avoid duplication of the same "entity" in the list of configurations
            if k not in config_default.keys():
                config_default[k] = skopt_space.dimensions[i].bounds[0]
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
        "logger": logger,
        "raise_exception_on_unsuitable_preprocessor": not no_exception_on_unsuitable_preprocessor
    }

    method_kwargs["run_function_kwargs"] = run_function_kwargs
    method_kwargs["callbacks"] = [TqdmCallback()] if verbose else []

    # Convert from MBs to Bytes
    memory_limit = workflow_memory_limit * (1024**2)
    memory_tracing_interval = 0.1
    raise_exception = False
    run_function = functools.partial(
        terminate_on_memory_exceeded,
        memory_limit,
        memory_tracing_interval,
        raise_exception,
        run_learning_workflow_from_deephyper,
    )

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


def main(**kwargs):
    """Entry point for the command line interface."""

    # setup logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger = logging.getLogger("LCDB")
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    # there is no point in making the logger configurable at the CLI, the log level maybe
    kwargs["logger"] = logger

    run_experiment(**kwargs)

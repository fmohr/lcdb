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
        "-c",
        "--campaign",
        type=str,
        required=False,
        default="default_campaign",
        help="The name of the campaign, which is mostly used for status files. Neither folders for configs nor for result files are inferred from",
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
        "-sd",
        "--status-dir",
        type=str,
        default="./exp-checkpoints",
        required=False,
        help="Directory where status files are managed.",
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
        "--parameters",
        type=str,
        default=None,
        required=False,
        help="JSON-parsable string with a dictionary that overwrites parameters specified in the configurations with constant values."
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

    subparser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        default="info",
        required=False,
        help="Controls the log level of the LCDB logger."
    )

    subparser.set_defaults(func=function_to_call)

def get_path_for_intermediate_results(campaign, workflow_class, openml_id, workflow_seed, test_seed, valid_seed, checkpoint_dir="./exp-checkpoints/"):
    import pathlib
    return pathlib.Path(f"{checkpoint_dir}/{campaign}/config-results/{workflow_class}/{openml_id}/{workflow_seed}-{test_seed}-{valid_seed}")

async def run_learning_workflow_from_deephyper_coroutine(job, **kwargs):
    return run_learning_workflow_from_deephyper(job, **kwargs)

def run_learning_workflow_from_deephyper(
        job,
        campaign: str = "",
        openml_id: int = 3,
        task_type: str = "classification",
        workflow_class: str = "lcdb.workflow.sklearn.LibLinearWorkflow",
        enforced_workflow_parameters: dict = {},
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
        memory_limit_in_bytes: int = 32 * (1024**3),  # 32 GB by default
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
    import json
    import pathlib
    from time import time

    if logger is None:
        logger = logging.getLogger("LCDB")

    # if this job has been done in the past, skip computations
    checkpoint_folder = get_path_for_intermediate_results(
        campaign=campaign,
        openml_id=openml_id,
        workflow_class=workflow_class,
        workflow_seed=workflow_seed,
        test_seed=test_seed,
        valid_seed=valid_seed
    )
    checkpoint_file_for_job = pathlib.Path(f"{checkpoint_folder}/{job.id}.json")
    if checkpoint_file_for_job.exists():
        with open(checkpoint_file_for_job, "r") as f:
            logger.info(f"Reading results for job {job.id} with parameters: {json.dumps(job.parameters)}.")
            return json.load(f)

    logger.info(f"Running job {job.id} with parameters: {json.dumps(job.parameters)}")

    import functools
    from lcdb import LCDB  # only to get the version of the code
    from lcdb.builder import run_learning_workflow
    from lcdb.builder.utils import terminate_on_memory_exceeded

    # Convert from MBs to Bytes
    memory_tracing_interval = 0.1
    log_interval = 5
    raise_exception = False
    run_function = functools.partial(
        terminate_on_memory_exceeded,
        memory_limit_in_bytes,
        memory_tracing_interval,
        raise_exception,
        run_learning_workflow,
        log_interval,
    )

    # add explicitly set parameters to the config (possibly overwriting configs specified in the config file)
    workfow_parameters = job.parameters.copy() # operate this on a copy so that deephyper isn't aware of the injected parameter values
    if enforced_workflow_parameters is not None:
        if type(enforced_workflow_parameters) != dict:
            raise ValueError(f"enforced_workflow_parameters should be a dict but is of type {type(enforced_workflow_parameters)}")
        workfow_parameters.update(enforced_workflow_parameters)

    # compute the learning curve
    t_start = time()
    
    results = run_function(
        openml_id=openml_id,
        task_type=task_type,
        workflow_class=workflow_class,
        workflow_parameters=workfow_parameters,
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
        epoch_schedule=epoch_schedule,
        memory_limit_in_bytes=memory_limit_in_bytes
    )
    t_end = time()

    # adding these results is important to avoid that the field is missing if the config is killed
    experiment_data = {
        "lcdb_version": LCDB.get_version(),
        "campaign": campaign,
        "openmlid": openml_id,
        "workflow_seed": workflow_seed,
        "workflow": workflow_class,
        "valid_prop": valid_prop,
        "test_prop": valid_prop,
        "monotonic": monotonic,
        "valid_seed": valid_seed,
        "test_seed": test_seed
    }
    if "metadata" in results:  # this is just for ordering purposes
        experiment_data.update(results["metadata"])
    results["metadata"] = experiment_data
    
    # set json to none if it is not there
    if "json" not in results["metadata"]:
        results["metadata"]["json"] = None

    checkpoint_file_for_job.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file_for_job, "w") as f:
        json.dump(results, f)
    return results


def run_experiment(
    campaign,
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
    status_dir,
    log_dir,
    max_evals,
    timeout,
    initial_configs,
    parameters,
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

    import pathlib

    import numpy as np
    import pandas as pd
    import json

    from deephyper.evaluator import Evaluator, HPOJob
    from deephyper.evaluator.callback import Callback, TqdmCallback
    from deephyper.evaluator.storage import MemoryStorage
    from deephyper.hpo import CBO, HpProblem
    from deephyper.hpo._problem import convert_to_skopt_space

    class JsonSanityCheckCallback(Callback):

        def on_done(self, job: HPOJob):
            logger.info(f"Checking sanity of metadata.")
            if job.metadata["json"] is None:
                logger.error("No JSON found in the output.")
                return
            try:
                json.loads(job.metadata["json"])
                logger.info(f"Done, detected proper and deserializable JSON. Proceeding.")
            except Exception as e:
                logger.exception(e)

    from lcdb.builder.utils import import_attr_from_module, StatusFileManager

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
    config_default = dict(config_space.get_default_configuration())
    
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
            config = row.to_dict()

            # replace nan values by default values, since deephyper cannot properly handle missing values
            for k, v in config.items():
                if type(v) == float and np.isnan(v):
                    config[k] = config_space[k].default_value # set nan values to default, should be ignored anyway
            initial_points.append(config)
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
        "campaign": campaign,
        "openml_id": openml_id,
        "task_type": task_type,
        "workflow_class": workflow_class,
        "enforced_workflow_parameters": parameters,
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
        "memory_limit_in_bytes": workflow_memory_limit * 1024**2,
        "raise_exception_on_unsuitable_preprocessor": not no_exception_on_unsuitable_preprocessor
    }

    method_kwargs["run_function_kwargs"] = run_function_kwargs
    method_kwargs["callbacks"] = [JsonSanityCheckCallback()]
    if verbose:
        method_kwargs["callbacks"].append(TqdmCallback())


    method_kwargs["storage"] = MemoryStorage()
    print("method_kwargs", method_kwargs)
    # print the run function setup

    # create status file manager
    status_file_manager = StatusFileManager(working_directory=status_dir)
    print(status_dir)
    
    with Evaluator.create(
        run_learning_workflow_from_deephyper if evaluator != "serial" else run_learning_workflow_from_deephyper_coroutine,
        method=evaluator,
        method_kwargs=method_kwargs,
    ) as evaluator:
        
        # check whether experiment has already run (master only checks)
        if evaluator.is_master:
            for status in [StatusFileManager.EXPERIMENT_STATUS_RUNNING, StatusFileManager.EXPERIMENT_STATUS_COMPLETED]:
                if status_file_manager.does_status_file_exist(workflow=workflow_class, campaign=campaign, openmlid=openml_id, workflowseed=workflow_seed, testseed=test_seed, valseed=valid_seed, status=status):
                    filename = status_file_manager.get_path_to_status_file(workflow=workflow_class, campaign=campaign, openmlid=openml_id, workflowseed=workflow_seed, testseed=test_seed, valseed=valid_seed, status=status)
                    logger.info(f"We have a status file {filename} for {workflow_class}-{campaign}-{openml_id}-{workflow_seed}-{test_seed}-{valid_seed} so the experiment is being skipped.")
                    return
                
            filename = status_file_manager.get_path_to_status_file(workflow=workflow_class, campaign=campaign, openmlid=openml_id, workflowseed=workflow_seed, testseed=test_seed, valseed=valid_seed, status=StatusFileManager.EXPERIMENT_STATUS_RUNNING)
            logger.info(f"Creating RUNNING status file {filename} for {workflow_class}-{campaign}-{openml_id}-{workflow_seed}-{test_seed}-{valid_seed}.")
            status_file_manager.create_status_file(workflow=workflow_class, campaign=campaign, openmlid=openml_id, workflowseed=workflow_seed, testseed=test_seed, valseed=valid_seed, status=StatusFileManager.EXPERIMENT_STATUS_RUNNING)

        # check whether we already have results in a results file
        list_of_previous_results = []
        num_previous_results = 0
        logger.info(f"Scanning folder {log_dir} folder already existing result files.")
        covered_indices = []
        for file in os.listdir(log_dir):
            if file.startswith("results") and file.endswith(".csv"):
                logger.debug(f"Reading results from {file}")
                df_results_in_file = pd.read_csv(f"{log_dir}/{file}")
                for i, row in df_results_in_file.iterrows():
                    config = {k[2:]: v for k, v in row.items() if k.startswith("p:")}
                    for idx, requested_config in enumerate(initial_points):
                        if config == requested_config and idx not in covered_indices:
                            covered_indices.append(idx)
                            list_of_previous_results.append(row)
                            num_previous_results += 1

        logger.info(f"Previous results found in {len(list_of_previous_results)} result files. Removing the {num_previous_results} configs with indices {covered_indices} from the todo list.")
        
        # Required for MPI just the root rank will execute the search
        # other ranks will be considered as workers
        if evaluator.is_master:

            # Set the search algorithm
            search = CBO(
                problem,
                evaluator,
                log_dir=log_dir,
                initial_points=[config for i, config in enumerate(initial_points) if i not in covered_indices],
                surrogate_model="DUMMY",
                verbose=verbose
            )

            # Execute the search (this will also generate/replace the results.csv)
            if num_previous_results > 0:
                max_evals -= num_previous_results
                assert max_evals > 0
            logger.info(f"Starting search with {max_evals} evaluations.")
            df_results_new = search.search(max_evals, timeout=timeout, max_evals_strict=True)
            if num_previous_results > 0:
                df_results = pd.concat([pd.DataFrame(list_of_previous_results), df_results_new])
            else:
                df_results = df_results_new
            out_file = f"{log_dir}/results.csv"
            logger.info(f"Writing {len(df_results)} results to {out_file}")
            df_results.to_csv(out_file, index=False)

            # now check whether we need to merge
            filename = status_file_manager.get_path_to_status_file(workflow=workflow_class, campaign=campaign, openmlid=openml_id, workflowseed=workflow_seed, testseed=test_seed, valseed=valid_seed, status=StatusFileManager.EXPERIMENT_STATUS_COMPLETED)
            logger.info(f"Creating COMPLETED status file {filename} for {workflow_class}-{campaign}-{openml_id}-{workflow_seed}-{test_seed}-{valid_seed}.")
            status_file_manager.create_status_file(
                workflow=workflow_class,
                campaign=campaign,
                openmlid=openml_id,
                workflowseed=workflow_seed,
                testseed=test_seed,
                valseed=valid_seed,
                status=StatusFileManager.EXPERIMENT_STATUS_COMPLETED
                )

            # remove checkpoint results if those exist
            folder_with_checkpoint_results = get_path_for_intermediate_results(
                    workflow_class=workflow_class,
                    campaign=campaign,
                    openml_id=openml_id,
                    workflow_seed=workflow_seed,
                    test_seed=test_seed,
                    valid_seed=valid_seed
                    )
            
            if folder_with_checkpoint_results.exists():
                import shutil
                shutil.rmtree(folder_with_checkpoint_results)
            


def main(**kwargs):
    """Entry point for the command line interface."""

    # setup logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger = logging.getLogger("LCDB")
    logger.handlers.clear()
    logger.addHandler(ch)

    accepted_log_levels = ["debug", "info", "warn", "error"]
    log_level = kwargs.pop("log_level")
    if log_level not in accepted_log_levels:
        raise ValueError(f"--log-level must be in {accepted_log_levels} but is {log_level}")
    if log_level == "debug":
        log_level = logging.DEBUG
    elif log_level == "info":
        log_level = logging.INFO
    elif log_level == "warn":
        log_level = logging.WARN
    elif log_level == "error":
        log_level = logging.ERROR

    ch.setLevel(log_level)
    logger.setLevel(log_level)

    # there is no point in making the logger configurable at the CLI, the log level maybe
    kwargs["logger"] = logger

    if "parameters" in kwargs and kwargs["parameters"] is not None:
        import json
        kwargs["parameters"] = json.loads(kwargs["parameters"])

    run_experiment(**kwargs)

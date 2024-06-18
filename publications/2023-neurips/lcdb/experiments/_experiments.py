import functools
import logging
import os
import pathlib
import pandas as pd
import shutil
import gzip


from .controller import run

class LCDB:

    @staticmethod
    def run_single_setup(
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

        import lcdb.json
        import pandas as pd
        from deephyper.evaluator import Evaluator, RunningJob
        from deephyper.evaluator.callback import TqdmCallback
        from deephyper.problem import HpProblem
        from deephyper.problem._hyperparameter import convert_to_skopt_space
        from deephyper.search.hps import CBO
        from .utils import import_attr_from_module, terminate_on_memory_exceeded


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
                raise ValueError(f"Specified file for initial configs {initial_configs} does not exist!")
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
            "logger": logger
        }

        method_kwargs["run_function_kwargs"] = run_function_kwargs
        method_kwargs["callbacks"] = [TqdmCallback()] if verbose else []

        # Convert from MBs to Bytes
        memory_limit = workflow_memory_limit * (1024 ** 2)
        memory_tracing_interval = 0.1
        raise_exception = False
        run_function = functools.partial(
            terminate_on_memory_exceeded,
            memory_limit,
            memory_tracing_interval,
            raise_exception,
            run,
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

    @staticmethod
    def run_campaign(
            campaign_dir,
            openml_ids,
            task_type,
            workflow_classes,
            monotonic,
            test_seeds,
            valid_seeds,
            workflow_seeds,
            valid_prop,
            test_prop,
            timeout_on_fit,
            log_dir,
            timeout,
            verbose,
            logger,
            evaluator,
            num_workers,
            anchor_schedule,
            epoch_schedule,
            workflow_memory_limit,
    ):

        # strip trailing slashes
        campaign_dir = campaign_dir.rstrip("/")

        # check that work directory exists and has a config file for the workflow
        if not pathlib.Path(campaign_dir).exists():
            raise ValueError(
                f"\n\n{'-' * 10}\n"
                f"Campaign directory {campaign_dir} does not exist.\n"
                f"{'-' * 10}\n"
                f"You need to create that directory and then place sub-directories for the workflows, in your case:\n"
                f"\t{campaign_dir}/{workflow_classes[0]}\n"
                f"Then, place a file configs.csv with the desired hyperparameter configurations of that workflow, i.e.,\n"
                f"\t{campaign_dir}/{workflow_classes[0]}/configs.csv"
            )

        for workflow_class in workflow_classes:
            if not pathlib.Path(f"{campaign_dir}/{workflow_class}").exists():
                raise ValueError(
                    f"\n\n{'-' * 10}\n"
                    f"Workflow directory {workflow_class} does not exist inside campaign directory {campaign_dir}.\n"
                    f"{'-' * 10}\n"
                    f"You need to create that directory and place a file configs.csv with the desired hyperparameter configurations of that workflow, i.e.,\n"
                    f"\t{campaign_dir}/{workflow_class}/configs.csv"
                )

        # run jobs
        # TODO: This needs to be parallelized more intelligently (probably)
        for workflow_class in workflow_classes:
            config_file = f"{campaign_dir}/{workflow_class}/configs.csv"
            if not os.path.exists(config_file):
                raise ValueError(f"File for initial_configs {config_file} does not exist!")
            num_configs = len(pd.read_csv(config_file))

            for openmlid in openml_ids:

                # create folder for results
                result_folder = f"{campaign_dir}/{workflow_class}/{openmlid}"
                pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

                for workflow_seed in workflow_seeds:
                    for test_seed in test_seeds:
                        for valid_seed in valid_seeds:

                            seed_str = f"{workflow_seed}-{test_seed}-{valid_seed}"

                            results_file_to = f"{result_folder}/{seed_str}.csv"
                            results_file_to_zipped = f"{results_file_to}.gz"
                            if pathlib.Path(results_file_to_zipped).exists():
                                print(f"Skipping existing file {results_file_to}")
                                continue

                            print(f"Processing {workflow_class} on dataset {openmlid} with seeds {seed_str}")

                            # run single setup
                            LCDB.run_single_setup(
                                openml_id=openmlid,
                                task_type=task_type,
                                workflow_class=workflow_class,
                                monotonic=monotonic,
                                valid_seed=valid_seed,
                                test_seed=test_seed,
                                workflow_seed=workflow_seed,
                                valid_prop=valid_prop,
                                test_prop=test_prop,
                                timeout_on_fit=timeout_on_fit,
                                log_dir=log_dir,
                                max_evals=num_configs,
                                timeout=timeout,
                                initial_configs=config_file,
                                verbose=verbose,
                                logger=logger,
                                evaluator=evaluator,
                                num_workers=num_workers,
                                anchor_schedule=anchor_schedule,
                                epoch_schedule=epoch_schedule,
                                workflow_memory_limit=workflow_memory_limit
                            )

                            # copy results file into the LCDB2 layout
                            results_file_from = f"{log_dir}/results.csv"
                            if not pathlib.Path(results_file_from).exists():
                                raise Exception(f"Did not find expected result files {results_file_from}.")
                            shutil.copy(results_file_from, results_file_to)
                            with open(results_file_to, 'rb') as f_in:
                                with gzip.open(results_file_to_zipped, 'wb', compresslevel=9) as f_out:
                                    shutil.copyfileobj(f_in, f_out,)
                            os.remove(results_file_to)

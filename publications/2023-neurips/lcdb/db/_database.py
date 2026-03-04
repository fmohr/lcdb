import json
import os
import pathlib
from statistics import mean
import trace
import time
import numpy as np
import pandas as pd
from lcdb.db.callbacks.counting_callback import CountingCallback
from lcdb.db._repository import Repository
from lcdb.db._results import ResultSet
from lcdb.db._util import get_path_to_lcdb,  CountAwareGenerator
from lcdb.builder.utils import convert_deephyper_result_row_to_dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from tqdm import tqdm
import logging

import pickle

def is_picklable(obj) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


logger = logging.getLogger("lcdb")

DETAIL_KEY = "results"

def _process_results(res, unpack_results, unpack_build_issues, processors):
    assert type(res) == list, f"Expected list of result rows but got {type(res)}"
    assert len(res) > 0, f"Expected non-empty list of result rows but got empty list."
    rs = ResultSet(res)
    if unpack_results:
        rs._unpack_results()
    if unpack_build_issues:
        rs._unpack_build_issues()
    if processors is not None:
        rs.apply(processors)
    return rs

class LCDB:
    """Used to represent the LCDB database.

    Args:
        path (str, optional): Path to the database folder that contains the LCDB config file. The config file is assumed to be name `.lcdb_config.json` by default, but this can be changed with the respective argument. In principle, this folder should be named ``.lcdb``, but this is not a requirement. Defaults to ``None`` that will first (1) check if a file with the name `config_filename` exists in the current working directory
        folder, if not it will (2) look if it exists in `~/.lcdb` where `~` is the home directory,
        if it is not in `~/.lcdb` (3) it will create it there as soon as an operation on the object is conducted (retrieval or aggregation of data).

        config_filename (str, optional): Name of the configuration file that is looked for.
    """

    def __init__(
        self,
        path: str = None
    ):

        # the following are now constants that are no longer configurable
        config_filename = "config.json"
        lcdb_folder = ".lcdb"

        # get path of LCDB
        self.path = pathlib.Path(
            get_path_to_lcdb() if path is None else f"{path}/{lcdb_folder}"
        )
        self.path_to_config = f"{self.path}/{config_filename}"
        self.path_to_count_table = f"{self.path}/counts.csv"

        # state vars
        self.loaded = False
        self._repositories = None

    def create(self, config=None):

        # create directory
        self.path.mkdir(exist_ok=True, parents=True)

        # create default config file
        default_config = {"repositories": {
            "official": "pcloud://kZeWywZRr6lScWSloHlzwk6Uxq3GyRtuBaX",
            "local": ".lcdb/data"}
        }
        if config is not None:
            default_config.update(config)
        config = default_config

        with open(f"{self.path_to_config}", "w") as f:
            json.dump(config, f)

    def exists(self):
        config_file = pathlib.Path(f"{self.path_to_config}")
        return config_file.exists()

    def _load(self):

        # check whether it exists
        if not self.exists():
            self.create()

        config_path = f"{self.path_to_config}"
        if not pathlib.Path(config_path).exists():
            raise Exception(
                f"LCDB at path {self.path.absolute()} seems corrupt. At least, it has no {self.config_filename}"
            )

        # read in config
        with open(config_path, "r") as f:
            cfg = json.load(f)
            repository_paths = {}
            for k, p in cfg["repositories"].items():
                if not p.startswith("pcloud://"):
                    p = os.path.expanduser(p)
                    if p[:1] != "/":
                        p = f"{self.path.parent}/{p}"
                repository_paths[k] = p

        self._repositories = {}
        for repository_name, repository_dir in repository_paths.items():
            self._repositories[repository_name] = Repository.get(repository_dir)

        self.loaded = True

    @classmethod
    def get_version(cls):

        # first get path to lcdb package
        path = pathlib.Path(__file__).parent.parent
        import subprocess

        # check whether this folder is tracked with git
        try:
            output = subprocess.check_output(
                ["git", "log", "-n", "1", "--pretty=format:%H", "--", str(path)],
                stderr=subprocess.DEVNULL
            )
            return output.decode().strip()

        except subprocess.CalledProcessError:
            from lcdb import __version__
            return __version__

    @property
    def repositories(self):
        if not self.loaded:
            self._load()
        return self._repositories
    
    @property
    def datasets(self):
        datasets = set()
        for repository in self.repositories.values():
            if repository.exists():
                datasets.update(repository.get_datasets(workflow=None, campaign=None))
        return sorted(datasets)
    
    @property
    def workflows(self):
        workflows = set()
        for repository in self.repositories.values():
            if repository.exists():
                workflows.update(repository.get_workflows())
        return sorted(workflows)
    
    def update_count_index(self):
        """
            Updates a local index file that contains information about the number of entries in the database
        """
        df = None
        for repo_name, repository in self.repositories.items():
            if repository.exists():
                df_repo = repository.get_count_table().copy()
                df_repo["repository"] = repo_name
                df = df_repo if df is None else pd.concat([df, df_repo], ignore_index=True)
        df = df[["repository", "workflow", "campaign", "openmlid", "seed_test", "seed_val", "seed_workflow", "num_configs"]]
        df.to_csv(self.path_to_count_table, index=False)

    @property
    def count_index(self):
        return pd.read_csv(self.path_to_count_table)

    def query(
            self,
            repositories=None,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            processors=None,
            unpack_results=True,
            unpack_build_issues=True,
            inclusion_predicate=None,
            max_workers=None,
            batch_size=100,
            buffer_size=2,
            callbacks=None
    ):
        """
        Gets a dictionary or generator of result dataframes. In the case of a dictionary, there is one dataframe per workflow; these are not unified since different workflows have different hyperparameters. In the case of a generator, each returned dataframe is for a single workflow, but it may (and typically will) occur that several dataframes for the same workflow are returned (but with values for different datasets or different seeds). In other words, it can always be assumed that the workflows of the returned dataframes (either by a generator or contained in the dictionary) have a homogenous worklfow attribute.

        Args:
            repositories (_type_, optional): _description_. Defaults to None.
            campaigns (_type_, optional): _description_. Defaults to None.
            workflows (_type_, optional): _description_. Defaults to None.
            openmlids (_type_, optional): _description_. Defaults to None.
            workflow_seeds (_type_, optional): _description_. Defaults to None.
            test_seeds (_type_, optional): _description_. Defaults to None.
            validation_seeds (_type_, optional): _description_. Defaults to None.
            show_progress (int, optional): _description_. Defaults to 0.

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        if not self.loaded:
            self._load()

        if repositories is None:
            repositories = list(self.repositories.values())
        else:
            requested_repository_names = set(repositories)
            existing_repository_names = set(self.repositories.keys())
            if (
                len(requested_repository_names.difference(existing_repository_names))
                > 0
            ):
                raise Exception(
                    f"The following repositories were included in the query but do not exist in this LCDB: "
                    f"{requested_repository_names.difference(existing_repository_names)}"
                )
            repositories = [self.repositories[k] for k in requested_repository_names]

        # make sure that required workflows are None or list
        if workflows is not None and isinstance(workflows, str):
            workflows = [workflows]

        # check that processors come in a list
        if processors is not None:
            if type(processors) != list:
                raise ValueError(f"processors must be None or list but are {type(processors)}")
            for p in processors:
                if not is_picklable(p):
                    raise ValueError(f"Processor {p} of type {type(p)} is not picklable, which is required for parallel processing. Please remove it from the processor list or make it picklable.")

        # create agenda
        df_agenda = self.count_index
        if campaigns is not None:
            df_agenda = df_agenda[df_agenda["campaign"].isin(campaigns)]
        if workflows is not None:
            df_agenda = df_agenda[df_agenda["workflow"].isin(workflows)]
        if openmlids is not None:
            df_agenda = df_agenda[df_agenda["openmlid"].isin(openmlids)]
        if test_seeds is not None:
            df_agenda = df_agenda[df_agenda["seed_test"].isin(test_seeds)]
        if validation_seeds is not None:
            df_agenda = df_agenda[df_agenda["seed_val"].isin(validation_seeds)]
        if workflow_seeds is not None:
            df_agenda = df_agenda[df_agenda["seed_workflow"].isin(workflow_seeds)]
        if inclusion_predicate is not None:
            df_agenda = df_agenda[df_agenda.apply(lambda row: inclusion_predicate(
                workflow=row["workflow"],
                openmlid=row["openmlid"],
                campaign=row["campaign"],
                workflow_seed=row["seed_workflow"],
                test_seed=row["seed_test"],
                val_seed=row["seed_val"]
            ), axis=1)]
        df_agenda = df_agenda.reset_index(drop=True).copy()

        # collect generators
        # result_generators = []
        # for repo_idx, repository in enumerate(repositories):
        #     if repository.exists():
        #         result_generators.append( 
        #             repository.query_results_as_stream(
        #                 campaigns=campaigns,
        #                 workflows=workflows,
        #                 openmlids=openmlids,
        #                 workflow_seeds=workflow_seeds,
        #                 test_seeds=test_seeds,
        #                 validation_seeds=validation_seeds,
        #                 inclusion_predicate=inclusion_predicate
        #             )
        #         )

        def generator():
            if max_workers is None or max_workers <= 1:
                num_workers = os.cpu_count()
            else:
                num_workers = max_workers
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = set()

                cur_result_set = None

                for workflow, df_agenda_workflow in df_agenda.groupby("workflow"):
                    for openmlid, df_agenda_workflow_and_dataset in df_agenda_workflow.groupby("openmlid"):
                        for repository_name, df_agenda_local in df_agenda_workflow_and_dataset.groupby("repository"):
                            
                            repository = self.repositories[repository_name]

                            # get generator for this query
                            gen = repository.query_results_as_stream(
                                campaigns=campaigns,
                                workflows=[workflow],
                                openmlids=[openmlid],
                                workflow_seeds=df_agenda_local["seed_workflow"].unique(),
                                test_seeds=df_agenda_local["seed_test"].unique(),
                                validation_seeds=df_agenda_local["seed_val"].unique()
                            )

                            for res in gen:

                                # check whether result is not None and put it into the queue for processing
                                assert res is not None
                                assert type(res) == list, f"Expected list of result rows but got {type(res)}"
                                assert len(res) > 0, f"Expected non-empty list of result rows but got empty list."

                                # if the queue is very full, wait for one to finish
                                if len(futures) >= buffer_size:
                                    wait(futures, return_when=FIRST_COMPLETED)
                                futures.add(executor.submit(_process_results, res, unpack_results, unpack_build_issues, processors))
                                num_futures_ready = sum(f.done() for f in futures)

                                # process futures that are ready
                                if num_futures_ready > 0:
                                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                                    for fut in done:
                                        if cur_result_set is not None:
                                            cur_result_set.extend(fut.result())
                                        else:
                                            cur_result_set = fut.result()
                                        
                                        # if the result set has the desired batch size, send results
                                        if len(cur_result_set) >= batch_size:
                                            to_deliver, cur_result_set = cur_result_set.split_at_index(batch_size)
                                            yield to_deliver
                        
                        # drain the remainings
                        while futures:
                            done, futures = wait(futures, return_when=FIRST_COMPLETED)
                            for fut in done:
                                if cur_result_set is not None:
                                    cur_result_set.extend(fut.result())
                                else:
                                    cur_result_set = fut.result()
                                
                                # if the result set has the desired batch size, send results
                                while len(cur_result_set) >= batch_size:
                                    to_deliver, cur_result_set = cur_result_set.split_at_index(batch_size)
                                    yield to_deliver
                        
                        # drain the rest anyway
                        if cur_result_set is not None and len(cur_result_set) > 0:
                            yield cur_result_set

                        # when the dataset is finished, send a callback
                        if callbacks is not None:
                            for cb in callbacks:
                                cb.on_workflow_dataset_combination_finished(workflow=workflow, openmlid=openmlid, total_num_records=df_agenda_workflow_and_dataset["num_configs"].sum())

        return CountAwareGenerator(int(np.ceil(df_agenda["num_configs"].sum() / batch_size)), generator())

    def statistics(
            self,
            repositories=None,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            show_progress=False,
            num_configs=None
    ):
        """
        Retrieves only rows that contain a traceback and their associated configs.

        Returns:
            pd.DataFrame: Execution and error statistics.
        """
        if not self.loaded:
            self._load()

        # Validate repositories
        if repositories is None:
            repositories = list(self.repositories.values())
        else:
            invalid_repos = set(repositories) - self.repositories.keys()
            if invalid_repos:
                raise ValueError(f"Invalid repositories: {invalid_repos}")
            repositories = [self.repositories[name] for name in repositories]

        if isinstance(workflows, str):
            workflows = [workflows]

        # Collect result generators with error handling
        result_generators = []
        for repo in repositories:
            if not repo.exists():
                print(f"Skipping non-existent repository: {repo}")
                continue  # Skip repositories that don't exist

            try:
                gen = repo.query_results_as_stream(
                    campaigns=campaigns,
                    workflows=workflows,
                    openmlids=openmlids,
                    workflow_seeds=workflow_seeds,
                    test_seeds=test_seeds,
                    validation_seeds=validation_seeds
                )

                if gen is None:
                    print(f"Warning: query_results_as_stream returned None for repository: {repo}")
                    continue  # Skip None results

                result_generators.append(gen)

            except Exception as e:
                print(f"Error retrieving results from {repo}: {e}")
                continue  # Skip on failure

        # Ensure we have valid generators
        if not result_generators:
            print("Error: No valid data sources found.")
            return pd.DataFrame()  # Return an empty DataFrame to avoid crashes

        # Create a generator function
        def generator():
            for gen in result_generators:
                try:
                    yield from gen
                except Exception as e:
                    print(f"Error processing generator: {e}")  # Log error and continue

        try:
            total_count = sum(len(g) for g in result_generators if hasattr(g, '__len__'))
            gen = CountAwareGenerator(total_count, generator())
        except TypeError:
            print("Error: One of the generators is invalid.")
            return pd.DataFrame()

        records = []  # Stores structured data for CSV output

 

        def process_traceback(df, num_configs=None):
            """Extracts traceback messages, errors, configs, execution times, and metadata."""
            if num_configs is None:
                num_configs = len(df)
            traceback_rows = df[df["m:traceback"].notna()]
            # instead of number of error measure number of non-nan m:json (they should represent the same thing)
            successfull_configs = df[df["m:json"].notna()]
            num_success_configs = len(successfull_configs)
            error_rate = (num_configs - num_success_configs) / num_configs if num_configs else 0

            tracebacks, errors = [], []
            for _, row in traceback_rows.iterrows():
                traceback_str = row["m:traceback"]
                try:
                    error_message = re.search(r'(\w+Error): (.*)', traceback_str).group(0)
                except AttributeError:
                    error_message = traceback_str  # Use full traceback if regex fails
                
                tracebacks.append(traceback_str)
                errors.append(error_message)

            # metadata extraction

            # get unique openmlids
            openmlids = df["m:openmlid"].unique()
            # dop nan values 
            openmlids = [x for x in openmlids if str(x) != 'nan']

            # get openmlid if it exists
            openmlid = openmlids[0] 

            if len(openmlids) > 1:
                print(f"Warning: Multiple OpenML IDs found in dataframe: {openmlids}")
            # get unique workflows
            workflows = df["m:workflow"].unique()
            # dop nan values
            workflows = [x for x in workflows if str(x) != 'nan']
            workflow = workflows[0] 

            if len(workflows) > 1:
                print(f"Warning: Multiple workflows found in dataframe: {workflows}")

            # convert to int
            if openmlid is not None:
                openmlid = int(openmlid)

            def get_mean_max_std(df, column_name):
                from scipy import stats

                # Convert to numeric and drop NaNs
                values = pd.to_numeric(df[column_name], errors="coerce").dropna()
                
                if not values.empty:
                    # mean
                    mean = values.mean()

                    # standard error and confidence interval
                    confidence = 0.95
                    n = len(values)
                    ci_low, ci_high = stats.t.interval(
                        confidence, df=n-1, loc=mean, scale=stats.sem(values)
                    )

                    # max
                    maximum = values.max()

                    return mean, ci_low, ci_high, maximum


            if "m:memory" in df.columns:
                mean_memory, ci_low_memory, ci_high_memory, max_memory = get_mean_max_std(df, "m:memory")
            else:
                mean_memory, ci_low_memory, ci_high_memory, max_memory = None, None, None, None

            if "m:timestamp_start" in df.columns and "m:timestamp_end" in df.columns:
                df["execution_time"] = df["m:timestamp_end"] - df["m:timestamp_start"]
                mean_config_time, ci_low_config_time, ci_high_config_time, max_config_time = get_mean_max_std(df, "execution_time")
            else:
                mean_config_time, ci_low_config_time, ci_high_config_time, max_config_time = None, None, None, None


            # append structured data
            records.append({
                "workflow": workflow,
                "openmlid": openmlid,
                "num_configs": num_configs,
                "error_rate": error_rate,
                "tracebacks": tracebacks,
                "errors": errors,
                "mean_memory": mean_memory,
                "ci_low_memory": ci_low_memory,
                "ci_high_memory": ci_high_memory,
                "max_memory": max_memory,
                "mean_config_time": mean_config_time,
                "ci_low_config_time": ci_low_config_time,
                "ci_high_config_time": ci_high_config_time,
                "max_config_time": max_config_time,
            })

        for df in tqdm(gen, disable=not show_progress):
            if df is not None and "m:traceback" in df.columns:
                process_traceback(df, num_configs)
            elif df is not None:
                print("Warning: No 'm:traceback' column in dataframe")

        result_df = pd.DataFrame(records)

        return result_df
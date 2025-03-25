import json
import os
import pathlib
import re
import trace
import pandas as pd
from lcdb.db._repository import Repository
from lcdb.db._util import get_path_to_lcdb,  CountAwareGenerator
from tqdm import tqdm


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

    @property
    def repositories(self):
        if not self.loaded:
            self._load()
        return self._repositories

    def query(
            self,
            repositories=None,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            return_generator=True,
            processors=None,
            show_progress=False
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

        result_generators = []
        for repository in repositories:
            if repository.exists():
                result_generators.append(
                    repository.query_results_as_stream(
                        campaigns=campaigns,
                        workflows=workflows,
                        openmlids=openmlids,
                        workflow_seeds=workflow_seeds,
                        test_seeds=test_seeds,
                        validation_seeds=validation_seeds,
                        processors=processors
                    )
                )

        def generator():
            for gen in result_generators:
                for res in gen:
                    yield res

        gen = CountAwareGenerator(sum([len(g) for g in result_generators]), generator())

        if return_generator:
            return gen
        else:
            dfs_per_workflow = {}
            for df in tqdm(gen, disable=not show_progress):
                workflow_class = df["m:workflow"].values[0]
                dfs_per_workflow[workflow_class] = df if workflow_class not in dfs_per_workflow else pd.concat([dfs_per_workflow[workflow_class], df])
            if workflows is not None and len(workflows) == 1:
                return dfs_per_workflow[workflows[0]] if workflows[0] in dfs_per_workflow else None
            else:
                return dfs_per_workflow
        
    
    def debug(
            self,
            repositories=None,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            show_progress=False
    ):
        """
        Retrieves only rows that contain a traceback and their associated configs.
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
                    f"The following repositories were included in the query but do not exist in this LCDB_debug: "
                    f"{requested_repository_names.difference(existing_repository_names)}"
                )
            repositories = [self.repositories[k] for k in requested_repository_names]

        if workflows is not None and isinstance(workflows, str):
            workflows = [workflows]

        result_generators = []
        for repository in repositories:
            if repository.exists():
                result_generators.append(
                    repository.query_results_as_stream(
                        campaigns=campaigns,
                        workflows=workflows,
                        openmlids=openmlids,
                        workflow_seeds=workflow_seeds,
                        test_seeds=test_seeds,
                        validation_seeds=validation_seeds,
                    )
                )

        def generator():
            for gen in result_generators:
                for res in gen:
                    yield res

        gen = CountAwareGenerator(sum([len(g) for g in result_generators]), generator())

        tracebacks, configs, errors = [], [], []

        for df in tqdm(gen, disable=not show_progress):

            if df is not None:
                # check if "traceback" column exists
                if "m:traceback" in df.columns:
                    traceback_rows = df[df["m:traceback"].notna()]

                    # print(traceback_rows)
                    for index, traceback_row in traceback_rows.iterrows():
                        traceback_str = traceback_row["m:traceback"]
                        traceback_frame = traceback_row.to_frame().T
                        traceback_indices = traceback_rows.index.tolist()
                        config_cols = [c for c in traceback_frame.columns if c.startswith("p:")]
                        corresponding_configs_reset = traceback_rows.loc[traceback_indices, config_cols].drop_duplicates().reset_index(drop=True)
                        configs.append(corresponding_configs_reset)
                        # extract errors from traceback messages str format first
                        try:
                            error_message = re.search(r'(\w+Error): (.*)', traceback_str).group(0)
                        except:
                            error_message = traceback_str
                        tracebacks.append(traceback_str)
                        errors.append(error_message)
                else:
                    print("Error: no traceback column in dataframe")

        return {
            "configs": pd.concat(configs, ignore_index=True) if configs else None,
            "tracebacks": pd.Series(tracebacks) if tracebacks else None,
            "errors": pd.Series(errors) if errors else None  
        }


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
            num_errors = len(traceback_rows)
            # instead of number of error measure number of non-nan m:json 
            successfull_configs = df[df["m:json"].notna()]
            num_success_configs = len(successfull_configs)
            error_rate = (num_configs - num_success_configs) / num_configs if num_configs else 0
            # error_rate = num_errors / num_configs if num_configs else 0

            tracebacks, errors = [], []
            for _, row in traceback_rows.iterrows():
                traceback_str = row["m:traceback"]
                try:
                    error_message = re.search(r'(\w+Error): (.*)', traceback_str).group(0)
                except AttributeError:
                    error_message = traceback_str  # Use full traceback if regex fails
                
                tracebacks.append(traceback_str)
                errors.append(error_message)

            # Extract metadata fields

            # get unique openmlids
            openmlids = df["m:openmlid"].unique()
            # dop nan values 
            # print if it has nans
            if any(pd.isnull(openmlids)):
                print(f"openmlids: {openmlids} has nans")
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

            # ensure openmlids column is integers
            if openmlid is not None:
                openmlid = int(openmlid)

            # take median and max memory usage
            if "m:memory" in df.columns:
                df["m:memory"] = pd.to_numeric(df["m:memory"], errors="coerce")  # Convert to numeric safely
                median_memory = df["m:memory"].median(skipna=True)  # Skip NaN values
                max_memory = df["m:memory"].max(skipna=True)

            if "m:timestamp_start" in df.columns and "m:timestamp_end" in df.columns:
                df["execution_time"] = df["m:timestamp_end"] - df["m:timestamp_start"]
                median_config_time = df["execution_time"].median()
                max_config_time = df["execution_time"].max()
            else:
                median_config_time, max_config_time = None, None

            # Append structured data
            records.append({
                "workflow": workflow,
                "openmlid": openmlid,
                "num_configs": num_configs,
                "error_rate": error_rate,
                "tracebacks": tracebacks,
                "errors": errors,
                "median_memory": median_memory,
                "max_memory": max_memory,
                "median_config_time": median_config_time,
                "max_config_time": max_config_time
            })

        for df in tqdm(gen, disable=not show_progress):
            if df is not None and "m:traceback" in df.columns:
                process_traceback(df, num_configs)
            elif df is not None:
                print("Warning: No 'm:traceback' column in dataframe")

        result_df = pd.DataFrame(records)

        return result_df

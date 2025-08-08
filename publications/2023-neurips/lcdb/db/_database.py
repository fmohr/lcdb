import json
import os
import pathlib
from statistics import mean
import trace
import pandas as pd
from lcdb.db._repository import Repository
from lcdb.db._util import get_path_to_lcdb,  CountAwareGenerator
from tqdm import tqdm
import logging


logger = logging.getLogger("lcdb")


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

    @classmethod
    def get_version(cls):

        # first get path to lcdb package
        path = pathlib.Path(__file__)
        import subprocess

        # check whether this folder is tracked with gut
        try:
            output = subprocess.check_output(
                ["git", "log", "-n", "1", "--pretty=format:%H", "--", str(path)],
                stderr=subprocess.DEVNULL
            )
            return output.decode().strip()

        except subprocess.CalledProcessError:
            return None

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
            logger.info("Creating dataframe from stream.")
            dfs_per_workflow = {}
            cnt = 0
            for df in tqdm(gen, disable=not show_progress):
                cnt += 1
                if df is None:
                    logger.warning("Received empty result dataframe.")
                    continue
                else:
                    workflow_class = df["m:workflow"].values[0]
                    dfs_per_workflow[workflow_class] = df if workflow_class not in dfs_per_workflow else pd.concat([dfs_per_workflow[workflow_class], df])
                    logger.debug(f"Added results from dataframe with {len(df)} entries. New length of dataframe for {workflow_class=} is {len(dfs_per_workflow[workflow_class])}")
            logger.info(f"Preparing results based on {cnt} seen dataframes for {len(dfs_per_workflow)} different workflows.")
            if workflows is not None and len(workflows) == 1:
                return dfs_per_workflow[workflows[0]] if workflows[0] in dfs_per_workflow else None
            else:
                return dfs_per_workflow

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
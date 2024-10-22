import json
import os
import pathlib
import re
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
            "official": "pcloud://kZK9f70Zxwwjkt54zA8FY6kBUFB5PXoAYT9k",
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
            # check if "traceback" column exists
            if "m:traceback" in df.columns:
                traceback_rows = df[df["m:traceback"].notna()]

                # extract corresponding configuration parameters
                if not traceback_rows.empty:
                    traceback_indices = traceback_rows.index.tolist()
                    config_cols = [c for c in df.columns if c.startswith("p:")]
                    # corresponding_configs = df.loc[traceback_rows.index]
                    # configs.append(corresponding_configs)
                    corresponding_configs_reset = df.loc[traceback_indices, config_cols].drop_duplicates().reset_index(drop=True)
                    configs.append(corresponding_configs_reset)

                    tracebacks.append(traceback_rows["m:traceback"])

                    # extract errors from traceback messages str format first
                    traceback_str = str(traceback_rows["m:traceback"].iloc[0])
                    try: 
                        error_message = re.search(r'(\w+Error): (.*)', traceback_str).group(0)
                    except:
                        error_message = traceback_str
                    errors.append(error_message)

            else:
                print("Error: no traceback column in dataframe")

        return {
            "configs": pd.concat(configs, ignore_index=True) if configs else None,
            "tracebacks": pd.concat(tracebacks, ignore_index=True) if tracebacks else None,
            "errors": pd.Series(errors) if errors else None  
        }
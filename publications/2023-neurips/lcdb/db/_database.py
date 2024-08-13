from ._util import get_path_to_lcdb
from ._repository import Repository
import pandas as pd
import json
import os
import pathlib


class LCDB:
    """Used to represent the LCDB database.

        Args:
            path (str, optional): Path to the database folder that contains the LCDB config file. The config file is assumed to be name `.lcdb_config.json` by default, but this can be changed with the respective argument. In principle, this folder should be named ``.lcdb``, but this is not a requirement.
            Defaults to ``None`` that will first (1) check if a file with the name `config_filename` exists in the current working directory
            folder, if not it will (2) look if it exists in `~/.lcdb` where `~` is the home directory,
            if it is not in `~/.lcdb` (3) it will create it there as soon as an operation on the object is conducted (retrieval or aggregation of data).

            config_filename (str, optional): Name of the configuration file that is looked for.
    """

    def __init__(self, path: str = None, config_filename: str = "config.json", lcdb_folder: str = ".lcdb"):

        # get path of LCDB
        self.path = pathlib.Path(get_path_to_lcdb() if path is None else f"{path}/{lcdb_folder}")
        self.path_to_config = f"{self.path}/{config_filename}"

        # state vars
        self.loaded = False
        self._repositories = None

    def create(self, config=None):

        # create directory
        self.path.mkdir(exist_ok=True, parents=True)

        # create default config file
        default_config = {
            "repositories": {
                "local": ".lcdb/data"
            }
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

    def get_results(
            self,
            repositories=None,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        if not self.loaded:
            self._load()

        if repositories is None:
            repositories = list(self.repositories.values())
        else:
            requested_repository_names = set(repositories)
            existing_repository_names = set(self.repositories.keys())
            if len(requested_repository_names.difference(existing_repository_names)) > 0:
                raise Exception(
                    f"The following repositories were included in the query but do not exist in this LCDB: "
                    f"{requested_repository_names.difference(existing_repository_names)}"
                )
            repositories = [self.repositories[k] for k in requested_repository_names]

        dfs = []
        for repository in repositories:
            if repository.exists():
                results_in_repo = repository.get_results(
                    campaigns=campaigns,
                    workflows=workflows,
                    openmlids=openmlids,
                    workflow_seeds=workflow_seeds,
                    test_seeds=test_seeds,
                    validation_seeds=validation_seeds
                )
                if results_in_repo is not None:
                    dfs.append(results_in_repo)
        if dfs:
            return pd.concat(dfs)
        else:
            return None

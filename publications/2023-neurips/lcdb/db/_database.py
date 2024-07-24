from ._util import get_database_location
from ._repository import Repository
import pandas as pd
import json
import os
import pathlib


class LCDB:

    def __init__(self, path=None):

        # get path of LCDB
        self.path = pathlib.Path(get_database_location() if path is None else path)
        self.loaded = False

        # state vars
        self._repositories = None

    def create(self, config=None):

        if path is None:
            path = pathlib.Path(lcdb_folder)
        path.mkdir(exist_ok=True, parents=True)

        default_config = {
            "repositories": {
                "home": "~/.lcdb/data",
                "local": ".lcdb/data"
            }
        }
        if config is not None:
            default_config.update(config)
        config = default_config

        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)

    def _load(self):

        # check whether it exists
        if not self.path.exists():
            raise Exception(f"Cannot load LCDB at path {self.path.absolute()}, which does not exist.")

        config_path = f"{self.path}/config.json"
        if not pathlib.Path(config_path).exists():
            raise Exception(f"LCDB at path {self.path.absolute()} seems corrupt. At least, it has no config.json")

        # read in config
        with open(config_path, "r") as f:
            cfg = json.load(f)
            repository_paths = {k: os.path.expanduser(p) for k, p in cfg["repositories"].items()}

        self._repositories = {}
        for repository_name, repository_dir in repository_paths.items():
            repository = Repository.get(repository_dir)
            if repository.exists():
                self._repositories[repository_name] = repository

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

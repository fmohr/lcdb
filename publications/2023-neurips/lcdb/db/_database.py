from ._util import get_repository_paths
from ._repository import Repository
import pandas as pd


class LCDB:

    def __init__(self):

        repository_paths = get_repository_paths()

        self.repositories = {}
        for repository_name, repository_dir in repository_paths.items():
            repository = Repository.get(repository_dir)
            if repository.exists():
                self.repositories[repository_name] = repository

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
            dfs.append(repository.get_results(
                campaigns=campaigns,
                workflows=workflows,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            ))
        if dfs:
            return pd.concat(dfs)
        else:
            return None

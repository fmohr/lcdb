import os

import pandas as pd
import gzip
from . import deserialize_dataframe


class ResultAggregator:

    CACHE_FOLDER = "~/.lcdb"

    def __init__(self, repos=["."]):
        super().__init__()

        self.repos = repos

    def add_repo(self, repo):
        if repo not in self.repos:
            self.repos.append(repo)

    @staticmethod
    def _get_subfolders(folder):
        return [f.name for f in os.scandir(folder) if f.is_dir()]

    @staticmethod
    def _get_experiment_campaigns_in_repo(repo):
        return ResultAggregator._get_subfolders(repo)

    @staticmethod
    def _get_result_files_in_folder(folder):
        return [f.name for f in os.scandir(folder) if f.is_file() and f.name.endswith((".csv", ".csv.gz"))]

    def get_results_for_all_configs(
            self,
            workflow,
            openmlid,
            workflow_seed=None,
            test_seed=None,
            validation_seed=None
    ):

        # get all result files
        result_files = []
        for repo in self.repos:
            for campaign in self._get_experiment_campaigns_in_repo(repo):
                folder_with_seeds = f"{repo}/{campaign}/{workflow}/{openmlid}"
                for result_file in ResultAggregator._get_result_files_in_folder(folder_with_seeds):
                    offset = 4 if result_file.endswith(".csv") else 7
                    _workflow_seed, _test_seed, _val_seed = [int(i) for i in result_file[:-offset].split("-")]
                    if workflow_seed is not None and workflow_seed != _workflow_seed:
                        continue
                    if test_seed is not None and test_seed != _test_seed:
                        continue
                    if validation_seed is not None and validation_seed != _val_seed:
                        continue

                    # append result file to the considered candidates
                    result_files.append(f"{folder_with_seeds}/{result_file}")

        # read in all result files
        dfs = []
        for file in result_files:
            if file.endswith((".gz", ".gzip")):
                with gzip.GzipFile(file, "rb") as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file)
            dfs.append(deserialize_dataframe(df))

        return pd.concat(dfs)

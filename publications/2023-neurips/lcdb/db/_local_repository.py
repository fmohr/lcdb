import os
from ._repository import Repository
import gzip
import pandas as pd
from ._dataframe import deserialize_dataframe
import pathlib


class LocalRepository(Repository):

    def __init__(self, repo_dir):
        super().__init__()
        self.repo_dir = repo_dir

    def read_result_file(self, file):
        if file.endswith((".gz", ".gzip")):
            with gzip.GzipFile(file, "rb") as f:
                df = pd.read_csv(f)  # , usecols=["p:bootstrap", "job_id", "m:traceback"])
        else:
            df = pd.read_csv(file)
        return df

    def add_results(self, campaign, *result_files):
        for result_file in result_files:
            df = self.read_result_file(result_file)
            for (workflow, openmlid, workflow_seed, valid_seed, test_seed), group in df.groupby(
                    ["m:workflow", "m:openmlid", "m:workflow_seed", "m:valid_seed", "m:test_seed"]
            ):
                folder = f"{self.repo_dir}/{workflow}/{campaign}/{openmlid}"
                pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
                print(f"made {folder}")
                filename = f"{folder}/{workflow_seed}-{test_seed}-{valid_seed}.csv.gz"
                group.to_csv(filename, index=False, compression='gzip')

    def get_workflows(self, campaign=None):
        raise NotImplementedError

    def get_datasets(self, campaign, workflow):
        raise NotImplementedError

    def get_result_files_of_workflow_and_dataset_in_campaign(
            self,
            workflow,
            campaign,
            openmlid,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        folder = f"{self.repo_dir}/{workflow}/{campaign}/{openmlid}"
        result_files_unfiltered = [
            f.name
            for f in os.scandir(folder)
            if f.is_file() and f.name.endswith((".csv", ".csv.gz"))
        ]

        result_files = []
        for filename in result_files_unfiltered:
            offset = 4 if filename.endswith(".csv") else 7
            _workflow_seed, _test_seed, _val_seed = [int(i) for i in filename[:-offset].split("-")]
            if workflow_seeds is not None and _workflow_seed not in workflow_seeds:
                continue
            if test_seeds is not None and _test_seed not in test_seeds:
                continue
            if validation_seeds is not None and _val_seed not in validation_seeds:
                continue

            result_files.append(f"{folder}/{filename}")
        return result_files

    def get_result_files_of_workflow_in_campaign(
            self,
            workflow,
            campaign,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        base_folder = f"{self.repo_dir}/{campaign}/{workflow}"
        if openmlids is None:
            openmlids = [f.name for f in os.scandir(base_folder) if f.is_dir()]

        filenames = []
        for openmlid in openmlids:
            filenames.extend(self.get_result_files_of_workflow_and_dataset_in_campaign(
                workflow=workflow,
                campaign=campaign,
                openmlid=openmlid,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            ))
        return filenames

    def get_result_files_of_workflow(
            self,
            workflow,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        base_folder = f"{self.repo_dir}/{workflow}"
        filenames = []
        if campaigns is None:
            campaigns = [f.name for f in os.scandir(base_folder) if f.is_dir()]
        for campaign in campaigns:
            filenames.extend(self.get_result_files_of_workflow_in_campaign(
                workflow=workflow,
                campaign=campaign,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            ))
        return filenames

    def get_results(
            self,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):

        # get all result files
        result_files = []

        if workflows is None:
            workflows = [f.name for f in os.scandir(self.repo_dir) if f.is_dir()]

        for workflow in workflows:
            result_files.extend(self.get_result_files_of_workflow(
                workflow=workflow,
                campaigns=campaigns,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            ))

        # read in all result files
        dfs = []
        for file in result_files:
            df = self.read_result_file(file)
            dfs.append(deserialize_dataframe(df))
        return pd.concat(dfs) if dfs else None

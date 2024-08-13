import os
import time

from ._repository import Repository
import gzip
import pandas as pd
from ._dataframe import deserialize_dataframe
import pathlib


class LocalRepository(Repository):

    def __init__(self, repo_dir):
        super().__init__()
        self.repo_dir = repo_dir

    def exists(self):
        return pathlib.Path(self.repo_dir).exists()

    def read_result_file(self, file, usecols=None):
        t_start = time.time()
        if file.endswith((".gz", ".gzip")):
            with gzip.GzipFile(file, "rb") as f:
                df = pd.read_csv(f, usecols=usecols)
        else:
            df = pd.read_csv(file, usecols=usecols)
        t_end = time.time()
        print(f"Reading {len(df)} lines with {df.shape[1]} cols from {file} took {int(1000 * (t_end - t_start))}ms.")
        return df

    def add_results(self, campaign, *result_files):
        for result_file in result_files:
            df = self.read_result_file(result_file)
            for (workflow, openmlid, workflow_seed, valid_seed, test_seed), group in df.groupby(
                    ["m:workflow", "m:openmlid", "m:workflow_seed", "m:valid_seed", "m:test_seed"]
            ):
                folder = f"{self.repo_dir}/{workflow}/{campaign}/{openmlid}"
                pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
                filename = f"{folder}/{workflow_seed}-{test_seed}-{valid_seed}.csv.gz"
                group.to_csv(filename, index=False, compression='gzip')

    def get_workflows(self):
        base_folder = self.repo_dir
        if not pathlib.Path(base_folder).exists():
            return []
        else:
            return [f.name for f in os.scandir(base_folder) if f.is_dir()]

    def get_campaigns(self, workflow):
        base_folder = f"{self.repo_dir}/{workflow}"
        if not pathlib.Path(base_folder).exists():
            return []
        else:
            return [f.name for f in os.scandir(base_folder) if f.is_dir()]

    def get_datasets(self, workflow, campaign):
        base_folder = f"{self.repo_dir}/{workflow}/{campaign}"
        if not pathlib.Path(base_folder).exists():
            return []
        else:
            return [f.name for f in os.scandir(base_folder) if f.is_dir()]

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

        if not pathlib.Path(folder).exists():
            return []

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

        if openmlids is None:
            openmlids = self.get_datasets(workflow=workflow, campaign=campaign)

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
        filenames = []
        if campaigns is None:
            campaigns = self.get_campaigns(workflow)
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

    def get_result_files(
            self,
            workflows=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        if workflows is None:
            workflows = self.get_workflows()

        result_files = []
        for workflow in workflows:
            result_files.extend(self.get_result_files_of_workflow(
                workflow=workflow,
                campaigns=campaigns,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            ))
        return result_files

    def get_num_results(
            self,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            result_files=None,
            max_cnt=10**6
    ):

        # get all files
        if result_files is None:
            result_files = self.get_result_files(
                workflows=workflows,
                campaigns=campaigns,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            )

        cnt = 0
        for f in result_files:
            cnt += len(self.read_result_file(f, usecols=["m:openmlid"]))
            if cnt >= max_cnt:
                return cnt
        return cnt

    def get_results(
            self,
            workflows=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):

        # get all result files
        result_files = self.get_result_files(
            workflows=workflows,
            campaigns=campaigns,
            openmlids=openmlids,
            workflow_seeds=workflow_seeds,
            test_seeds=test_seeds,
            validation_seeds=validation_seeds
        )

        if self.get_num_results(result_files=result_files) > 10**6:
            raise ValueError(f"Cannot read in more than 10**6 results.")

        # read in all result files
        dfs = []
        for file in result_files:
            df = self.read_result_file(file)
            t_before = time.time()
            dfs.append(deserialize_dataframe(df))
            t_after = time.time()
            print(f"Appending deserialized dataframe took {int(1000 * (t_after - t_before))}ms")
        return pd.concat(dfs) if dfs else None

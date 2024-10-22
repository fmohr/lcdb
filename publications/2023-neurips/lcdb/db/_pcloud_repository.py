from ._util import CountAwareGenerator

import gzip
import logging
import io
import time

import pandas as pd

from lcdb.db._dataframe import deserialize_dataframe
from lcdb.db._repository import Repository

import requests
import jmespath


class PCloudRepository(Repository):

    def __init__(self, repo_code):
        super().__init__()
        self.repo_code = repo_code
        response = requests.get(f"https://api.pcloud.com/showpublink?code={self.repo_code}")
        self.content = response.json()

    def exists(self):
        return self.content is not None and len(self.content) > 0

    def read_result_file(self, file, usecols=None):

        # get download link
        response = requests.get(f"https://api.pcloud.com/getpublinkdownload?code={self.repo_code}&fileid={file}").json()
        download_link = "https://" + response["hosts"][0] + response["path"]

        # download file
        response = requests.get(download_link)
        if response.status_code == 200:

            t_start = time.time()
            if download_link.endswith((".gz", ".gzip")):
                compressed_file = io.BytesIO(response.content)
                with gzip.GzipFile(fileobj=compressed_file) as f:
                    df = pd.read_csv(f, usecols=usecols)
            else:
                df = pd.read_csv(file, usecols=usecols)
            t_end = time.time()
            logging.info(
                f"Reading {len(df)} lines with {df.shape[1]} cols from {file} took {int(1000 * (t_end - t_start))}ms.")
            return df

        else:
            print(f"Failed to fetch the file. Status code: {response.status_code}")

    def add_results(self, campaign, *result_files):
        raise NotImplementedError

    def get_workflows(self):
        return jmespath.compile("metadata.contents[? name == 'data'] | [0] .contents | [*].name").search(self.content)

    def get_campaigns(self, workflow):
        return jmespath.compile(
            f"""
            metadata
            .contents[? name == 'data'] | [0]
            .contents | [? name == '{workflow}'] | [0]
            .contents | [*].name"""
        ).search(self.content)

    def get_datasets(self, workflow, campaign):
        return sorted([
            int(i) for i in jmespath.compile(
                f"""
                metadata
                .contents[? name == 'data'] | [0]
                .contents | [? name == '{workflow}'] | [0]
                .contents | [? name == '{campaign}'] | [0]
                .contents | [*].name
                """
            ).search(self.content)]
        )

    def get_result_files_of_workflow_and_dataset_in_campaign(
            self,
            workflow,
            campaign,
            openmlid,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        result_files_unfiltered = jmespath.compile(
            f"""
                metadata.contents[? name == 'data'] | [0]
                .contents | [? name == '{workflow}'] | [0]
                .contents | [? name == '{campaign}'] | [0]
                .contents | [? name == '{openmlid}'] | [0]
                .contents | [*]
            """).search(self.content)

        # now collect file ids of matching files
        result_files = []
        for file_data in result_files_unfiltered:
            filename = file_data["name"]
            offset = 4 if filename.endswith(".csv") else 7
            try:
                _workflow_seed, _test_seed, _val_seed = [int(i) for i in filename[:-offset].split("-")]
                if workflow_seeds is not None and _workflow_seed not in workflow_seeds:
                    continue
                if test_seeds is not None and _test_seed not in test_seeds:
                    continue
                if validation_seeds is not None and _val_seed not in validation_seeds:
                    continue
            except ValueError:
                print(f"Invalid filename {filename}")
                continue

            result_files.append(file_data["fileid"])
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

    def query_results_as_stream(
            self,
            workflows=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            processors=None
    ):
        """

        :param workflows: iterable of workflow names for which results are desired (None for all available)
        :param campaigns: iterable of campaign names from which results are desired (None for all available)
        :param openmlids: iterable of datasets (integers) for which results are desired (None for all available)
        :param workflow_seeds: iterable of workflow seeds (integers) for which results are desired (None for all available)
        :param test_seeds: iterable of dataset test split seeds (integers) for which results are desired (None for all available)
        :param validation_seeds: iterable of dataset validation split seeds (integers) for which results are desired (None for all available)
        :return:
        """

        if processors is not None and not isinstance(processors, dict):
            raise ValueError(f"processors must be None or a dictionary with Callables as values.")

        # get all result files
        result_files = self.get_result_files(
            workflows=workflows,
            campaigns=campaigns,
            openmlids=openmlids,
            workflow_seeds=workflow_seeds,
            test_seeds=test_seeds,
            validation_seeds=validation_seeds
        )

        # read in all result files

        def gen_fun():
            total_entries = 0

            for file in result_files:
                if total_entries > 10 ** 6:
                    raise ValueError(f"Cannot read in more than 10**6 results.")
                df = self.read_result_file(file)
                df_deserialized = deserialize_dataframe(df)
                if processors is not None:
                    for name, fun in processors.items():
                        df[name] = df.apply(fun, axis=1)  # apply the function to all rows in the dataframe
                    df.drop(columns="m:json", inplace=True)

                total_entries += len(df_deserialized)
                yield df_deserialized

        return CountAwareGenerator(len(result_files), gen=gen_fun())

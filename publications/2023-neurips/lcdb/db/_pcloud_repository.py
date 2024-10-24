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

    def __init__(self, repo_code, token=None):
        super().__init__()
        self.repo_code = repo_code
        self.content = None
        self.token = token

        # update content
        self.update_content()

    def update_content(self):
        self.content = requests.get(f"https://api.pcloud.com/showpublink?code={self.repo_code}").json()

    def exists(self):
        return self.content is not None and len(self.content) > 0

    def authenticate(self, username, password):
        url = f"https://api.pcloud.com/userinfo?getauth=1&logout=1&device=lcdbclient"
        response = requests.post(url, {
            "username": username,
            "password": password
        }).json()
        self.token = response["auth"] if "auth" in response else None
        if self.token is None:
            raise ValueError(f"Authentication failed. Response from server was {response}.")

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

    def _get_folder_id(self, workflow=None, campaign=None, openmlid=None):
        """
            Returns the folderid of the folder containing the data for this context.
            Returns None if that folder does not exist
        """

        # create query
        query = "metadata .contents[? name == 'data'] | [0]"
        if workflow is not None:
            query += f".contents | [? name == '{workflow}'] | [0]"
            if campaign is not None:
                query += f".contents | [? name == '{campaign}'] | [0]"
                if openmlid is not None:
                    query += f".contents | [? name == '{openmlid}'] | [0]"

            elif openmlid is not None:
                raise ValueError("openmlid can be only set if both a workflow and a campaign are given.")

        elif campaign is not None or openmlid is not None:
            raise ValueError("campaign or openmlid can be only set if a workflow is given.")

        return jmespath.compile(f"{query}.folderid").search(self.content)

    def _create_folder(self, parent_folder_id, name):
        response = requests.get(
            f"https://api.pcloud.com/createfolder?code={self.repo_code}&auth={self.token}&folderid={parent_folder_id}&name={name}"
        ).json()
        if response is None:
            raise ValueError(f"Could not create folder '{name}', received no response")
        if "result" not in response or response["result"] != 0:
            raise ValueError(f"Could not create folder '{name}', received invalid response: {response}")
        self.update_content()
        return response["metadata"]["folderid"]

    def add_results(self, campaign, *result_files):
        self.update_content()  # make sure that we have the current file structure at pCloud
        for result_file in result_files:

            # read result file
            if result_file.endswith((".gz", ".gzip")):
                with gzip.GzipFile(result_file, "rb") as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(result_file)

            # decompose this dataframe so that we have results only for a single workflow/openmlid and seeds
            for (workflow, openmlid, workflow_seed, valid_seed, test_seed), group in df.groupby(
                    ["m:workflow", "m:openmlid", "m:workflow_seed", "m:valid_seed", "m:test_seed"]
            ):
                name = f"{int(workflow_seed)}-{int(test_seed)}-{int(valid_seed)}.csv.gz"
                print(f"Adding results for {workflow}/{campaign}/{openmlid}/{name}")
                folder_id = self._get_folder_id(workflow=workflow, campaign=campaign, openmlid=openmlid)

                # if the folder does not exist, create one
                if folder_id is None:

                    folder_id_root = self._get_folder_id()

                    # create workflow folder if necessary
                    folder_id_workflow = self._get_folder_id(workflow=workflow)
                    if folder_id_workflow is None:
                        print(f"create workflow folder {workflow} in folder id {folder_id_root}")
                        folder_id_workflow = self._create_folder(parent_folder_id=folder_id_root, name=workflow)

                    # create campaign folder if necessary
                    folder_id_campaign = self._get_folder_id(workflow=workflow, campaign=campaign)
                    if folder_id_campaign is None:
                        print(f"create campaign folder inside {folder_id_workflow}")
                        folder_id_campaign = self._create_folder(parent_folder_id=folder_id_workflow, name=campaign)

                    # create dataset folder if necessary
                    folder_id_dataset = self._get_folder_id(workflow=workflow, campaign=campaign, openmlid=openmlid)
                    if folder_id_dataset is None:
                        print(f"create dataset folder inside {folder_id_campaign}")
                        folder_id = self._create_folder(parent_folder_id=folder_id_campaign, name=openmlid)

                # Create a BytesIO object to hold the CSV in binary format
                csv_buffer = io.BytesIO()

                # Write the DataFrame to the buffer in CSV format, but use StringIO first to handle text conversion
                csv_string = df.to_csv(index=False)

                # Compress the CSV data using gzip
                with gzip.GzipFile(fileobj=csv_buffer, mode='wb') as gz:
                    gz.write(csv_string.encode('utf-8'))  # Compress the CSV string (convert it to bytes first)

                # Reset the buffer's position to the beginning
                csv_buffer.seek(0)

                # upload the file
                url = f"https://api.pcloud.com/uploadfile?code={self.repo_code}&auth={self.token}&folderid={folder_id}&filename={name}"
                status = requests.post(url, files={'file': (name, csv_buffer, 'application/gzip')}).json()
                if not isinstance(status, dict):
                    raise ValueError(
                        f"Could not add result. Object received from pCloud should be a dict but is {type(status)}"
                    )
                if status["result"] != 0:
                    raise ValueError(f"Could not add result. Received an error response from pCloud: {status}")

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
            workflow=None,
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

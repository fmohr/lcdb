from turtle import update
from ._util import CountAwareGenerator

import gzip
import logging
import io
import time
import os
import json
import jsonlines

import pandas as pd
import numpy as np

from lcdb.db._repository import Repository
from lcdb.builder.utils import convert_deephyper_result_row_to_dict


import requests
import jmespath
from json import JSONDecodeError


class PCloudRepository(Repository):

    def __init__(self, repo_code, token=None):
        super().__init__()
        self.repo_code = repo_code
        self.content = None
        self.token = token
        # update content
        self.update_content()
        self.root_folder_id = self.content['metadata'].get('folderid')

    def update_content(self):
        self.content = requests.get(f"https://eapi.pcloud.com/showpublink?code={self.repo_code}").json()


    def exists(self):
        return self.content is not None and len(self.content) > 0

    def authenticate(self, username, password, device="lcdbclient", authexpire=300):
        """

        :param username: the username at pCloud with access to this repository
        :param password: the password of the user
        :param device: name associated with the authentication request (usually no change reasonable)
        :param authexpire: time in seconds after which the received token will expire
        :return:
        """
        url = f"https://eapi.pcloud.com/userinfo?getauth=1&logout=1&device={device}&authexpire={authexpire}"
        response = requests.post(url, {
            "username": username,
            "password": password
        }).json()
        self.token = response["auth"] if "auth" in response else None
        if self.token is None:
            raise ValueError(f"Authentication failed. Response from server was {response}.")

    def read_result_file(self, file):

        # get download link
        response = requests.get(f"https://eapi.pcloud.com/getpublinkdownload?code={self.repo_code}&fileid={file}").json()
        download_link = "https://" + response["hosts"][0] + response["path"]

        # download file
        t_start = time.time()
        logging.info(f"Starting download of {download_link}")
        response = requests.get(download_link)
        t_end_dl = time.time()
        logging.info(f"Download finished after {int(1000 * (t_end_dl - t_start))} miliseconds")
        if response.status_code == 200:

            t_start = time.time()
            if download_link.endswith((".gz", ".gzip")):
                filehandle = io.BytesIO(response.content)
            else:
                filehandle = download_link
            with gzip.open(filehandle, 'rt', encoding='utf-8') as f:
                reader = jsonlines.Reader(f)
                for obj in reader:
                    yield obj
        else:
            print(f"Failed to fetch the file. Status code: {response.status_code}")

    def _get_folder_id(self, path=None, root=False):
        """Returns the folder ID for a specified full path within the repository."""
        if root:
            # return the "data" folder ID
            return jmespath.compile("metadata.contents[? name == 'data'] | [0] .folderid").search(self.content)
        if path is None:
            raise ValueError("Path must be specified to get a folder ID.")

        parts = path.split('/')  # Split the path into parts: [dataset, model_size, anchor]
        folder_id = self.root_folder_id

        for part in parts:
            query = f"metadata.contents[?name=='{part}'] | [0].folderid"
            folder_id = jmespath.compile(query).search(self.content)
            if folder_id is None:
                return None  # Folder not found
            self.content = requests.get(
                f"https://eapi.pcloud.com/listfolder?code={self.repo_code}&auth={self.token}&folderid={folder_id}"
            ).json()  # Update content to current folder

        return folder_id

    def _create_folder(self, parent_folder_id, name):
        self.update_content()
        response = requests.get(
            f"https://eapi.pcloud.com/createfolder?code={self.repo_code}&auth={self.token}&folderid={parent_folder_id}&name={name}"
        ).json()
        if response is None:
            raise ValueError(f"Could not create folder '{name}', received no response")
        if "result" not in response or response["result"] != 0:
            raise ValueError(f"Could not create folder '{name}', received invalid response: {response}")
        self.update_content()
        return response["metadata"]["folderid"]
    

    def _get_or_create_folder_id(self, full_path):
        """Gets or creates the folder ID for the specified path."""
        self.update_content()
        print(f"Updated content: {self.content}")  # Debugging

        parts = full_path.split("/")
        parent_folder_id = self.root_folder_id
        print(f"Root folder ID: {parent_folder_id}")  # Debugging

        for part in parts:
            print(f"Checking folder: '{part}' in parent {parent_folder_id}")  # Debugging

            # Fetch folder contents for the current parent folder ID
            response = requests.get(
                f"https://eapi.pcloud.com/listfolder?code={self.repo_code}&auth={self.token}&folderid={parent_folder_id}"
            ).json()
            print(f"API Response: {response}")  # Debugging

            # Check for error in the response
            if response.get("result") != 0:
                print(f"Error fetching folder contents: {response.get('error')}")
                return None

            folder = next((item for item in response.get("metadata", {}).get("contents", [])
                           if item["name"] == part and item["isfolder"]), None)

            if folder:
                parent_folder_id = folder["folderid"]
            else:
                parent_folder_id = self._create_folder(parent_folder_id, part)
                

        # If all parts are found, return the final parent_folder_id
        return parent_folder_id


    
    def add_results(self, campaign, *result_files):
        """Uploads result files (JSONL or CSV) to pCloud, preserving lcdb/data/<workflow>/<campaign>/<openmlid>/<file> structure."""
        import gzip
        import io
        import os
        import re
        import json
        import requests
        import pandas as pd

        self.update_content()

        for result_file in result_files:
            is_jsonl = result_file.endswith((".jsonl", ".jsonl.gz", ".jsonl.gzip"))

            # ------------------------------------------------------------
            # Extract workflow and openmlid from file path
            # ------------------------------------------------------------
            workflow = None
            openmlid = None

            parts = result_file.split("/")
            for part in parts:
                if part.startswith("lcdb.workflow"):
                    workflow = part.split("-")[0]  # strip memory bin suffix
                if re.fullmatch(r"\d+", part):
                    openmlid = part

            if workflow is None:
                workflow = "unknown_workflow"
            if openmlid is None:
                openmlid = "unknown_dataset"

            # ------------------------------------------------------------
            # Derive seeds from JSONL content (if available)
            # ------------------------------------------------------------
            workflow_seed = valid_seed = test_seed = 0

            if is_jsonl:
                opener = gzip.open if result_file.endswith((".gz", ".gzip")) else open
                try:
                    with opener(result_file, "rt", encoding="utf-8") as f:
                        first_line = f.readline()
                        if first_line.strip():
                            record = json.loads(first_line)
                            workflow_seed = int(record.get("workflow_seed", 0))
                            valid_seed = int(record.get("valid_seed", 0))
                            test_seed = int(record.get("test_seed", 0))
                except Exception as e:
                    print(f"Warning: Could not extract seeds from {result_file}: {e}")

            # ------------------------------------------------------------
            # Build destination path and base filename
            # ------------------------------------------------------------
            path = f"data/{workflow}/{campaign}/{openmlid}"
            folder_id = self._get_or_create_folder_id(path)

            if is_jsonl:
                base_name = f"{workflow_seed}-{test_seed}-{valid_seed}.jsonl.gz"
            else:
                base_name = f"{workflow_seed}-{test_seed}-{valid_seed}.csv.gz"

            print(f"Uploading {base_name} to {path}")

            # ------------------------------------------------------------
            # JSONL(.gz): upload raw file as-is
            # ------------------------------------------------------------
            if is_jsonl:
                if result_file.endswith((".gz", ".gzip")):
                    with open(result_file, "rb") as f:
                        content = f.read()
                else:
                    with open(result_file, "rb") as f:
                        data = f.read()
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                        gz.write(data)
                    buf.seek(0)
                    content = buf.read()

                upload_buf = io.BytesIO(content)
                url = (
                    f"https://eapi.pcloud.com/uploadfile"
                    f"?code={self.repo_code}&auth={self.token}"
                    f"&folderid={folder_id}&filename={base_name}"
                )
                status = requests.post(
                    url, files={"file": (base_name, upload_buf, "application/gzip")}
                ).json()

            # ------------------------------------------------------------
            # CSV(.gz): legacy fallback
            # ------------------------------------------------------------
            else:
                if result_file.endswith((".gz", ".gzip")):
                    with gzip.open(result_file, "rt", encoding="utf-8") as f:
                        df = pd.read_csv(f)
                else:
                    df = pd.read_csv(result_file)

                csv_buffer = io.BytesIO()
                csv_string = df.to_csv(index=False)
                with gzip.GzipFile(fileobj=csv_buffer, mode="wb") as gz:
                    gz.write(csv_string.encode("utf-8"))
                csv_buffer.seek(0)

                url = (
                    f"https://eapi.pcloud.com/uploadfile"
                    f"?code={self.repo_code}&auth={self.token}"
                    f"&folderid={folder_id}&filename={base_name}"
                )
                status = requests.post(
                    url, files={"file": (base_name, csv_buffer, "application/gzip")}
                ).json()

            # ------------------------------------------------------------
            # Verify upload result
            # ------------------------------------------------------------
            if not isinstance(status, dict):
                raise ValueError(
                    f"Could not add result. pCloud returned {type(status)} instead of dict."
                )
            if status.get("result") != 0:
                raise ValueError(
                    f"pCloud upload failed for {result_file}: {status}"
                )                


    def _upload_file(self, folder_id, file_path, file_name):
        """Uploads a file to a specific pCloud folder."""
        url = f"https://eapi.pcloud.com/uploadfile?code={self.repo_code}&auth={self.token}&folderid={folder_id}&filename={file_name}"
        with open(file_path, "rb") as f:
            response = requests.post(url, files={"file": (file_name, f)}).json()
            if response.get("result") != 0:
                raise ValueError(f"Failed to upload file '{file_name}': {response}")

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
        qry = f"""
                metadata
                .contents[? name == 'data'] | [0]
            """
        if workflow is not None:
            qry += f".contents | [? name == '{workflow}'] | [0]"
        else:
            qry += ".contents | [*] | [0]"
        
        if campaign is not None:
            qry += f".contents | [? name == '{campaign}'] | [0]"
        else:
            qry += ".contents | [*] | [0]"
        qry += ".contents | [*].name"
        qry_result = jmespath.compile(qry).search(self.content)
        if qry_result is None:
            return []
        return sorted([int(i) for i in qry_result])

    def get_result_files_of_workflow_and_dataset_in_campaign(
            self,
            workflow,
            campaign,
            openmlid,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        # print(f"Getting result files for {workflow}/{campaign}/{openmlid}")
        result_files_unfiltered = jmespath.compile(
            f"""
                metadata.contents[? name == 'data'] | [0]
                .contents | [? name == '{workflow}'] | [0]
                .contents | [? name == '{campaign}'] | [0]
                .contents | [? name == '{openmlid}'] | [0]
                .contents | [*]
            """).search(self.content)

        # If result_files_unfiltered is None, assign an empty list
        if result_files_unfiltered is None:
            result_files_unfiltered = []

        # now collect file ids of matching files
        result_files = []
        for file_data in result_files_unfiltered:
            filename = file_data["name"]
            offset = 6 if filename.endswith(".jsonl") else 9
            try:
                _workflow_seed, _test_seed, _val_seed = [int(i) for i in filename[:-offset].split("-")]
                if workflow_seeds is not None and _workflow_seed not in workflow_seeds:
                    continue
                if test_seeds is not None and _test_seed not in test_seeds:
                    continue
                if validation_seeds is not None and _val_seed not in validation_seeds:
                    continue
                if openmlid is not None and openmlid != int(openmlid):
                    continue
            except ValueError:
                print(f"Could not load file {filename} from pCloud repository. Invalid filename {filename}")
                continue

            result_files.append([workflow, campaign, openmlid, _workflow_seed, _test_seed, _val_seed, file_data["fileid"]])
        return pd.DataFrame(result_files, columns=["workflow", "campaign", "openmlid", "seed_workflow", "seed_test", "seed_val", "fileid"])

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

        result_files = None
        for openmlid in openmlids:
            result_files_new = self.get_result_files_of_workflow_and_dataset_in_campaign(
                workflow=workflow,
                campaign=campaign,
                openmlid=openmlid,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            )
            result_files = result_files_new if result_files is None else pd.concat([result_files, result_files_new])
        return result_files

    def get_result_files_of_workflow(
            self,
            workflow=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        result_files = None
        if campaigns is None:
            campaigns = self.get_campaigns(workflow)
        for campaign in campaigns:
            result_files_new = self.get_result_files_of_workflow_in_campaign(
                workflow=workflow,
                campaign=campaign,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            )
            result_files = result_files_new if result_files is None else pd.concat([result_files, result_files_new])
        return result_files

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

        result_files = None
        for workflow in workflows:
            result_files_new = self.get_result_files_of_workflow(
                workflow=workflow,
                campaigns=campaigns,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds
            )
            result_files = result_files_new if result_files is None else pd.concat([result_files, result_files_new])
        return result_files

    def query_results_as_stream(
            self,
            workflows=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            raise_errors=False,
            report_errors=True
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
        def gen_fun(raise_errors=False):
            total_entries = 0
            if result_files is None:
                return
            for i, file_desc in result_files.iterrows():
                if total_entries > 10 ** 6:
                    raise ValueError(f"Cannot read in more than 10**6 results.")
 
                try:
                    rows = [convert_deephyper_result_row_to_dict(row) for row in self.read_result_file(file_desc["fileid"])]
                    total_entries += len(rows)
                    yield rows

                except Exception as e:
                    is_parsing_error = isinstance(e, JSONDecodeError)
                    if is_parsing_error:
                        error_msg = f"Parsing error {repr(e)} for result file:"
                    else:
                        error_msg = f"{type(e)} with message '{repr(e)}' in result file:"

                    error_msg += ""\
                                f"\n\tworkflow {file_desc['workflow']}"\
                                f"\n\tcampaign {file_desc['campaign']}"\
                                f"\n\topenmlid {file_desc['openmlid']}"\
                                f"\n\tseed_wf {file_desc['seed_workflow']}"\
                                f"\n\tseed_test {file_desc['seed_test']}"\
                                f"\n\tseed_valid {file_desc['seed_val']}"\
                                f"\n\tpCloud file id {file_desc['fileid']}"
                    if raise_errors:
                        raise Exception(error_msg)
                    elif report_errors:
                        print(error_msg)
                    df = None

        return CountAwareGenerator(len(result_files) if result_files is not None else 0, gen=gen_fun())

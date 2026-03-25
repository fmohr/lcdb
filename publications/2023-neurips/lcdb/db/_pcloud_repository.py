from turtle import update
from ._util import CountAwareGenerator

import gzip
import logging
import io
import os
import re
import time
import json
import jsonlines
import zipfile
import tempfile

import pandas as pd
import numpy as np
from tqdm import tqdm

from lcdb.db._repository import Repository
from lcdb.builder.utils import convert_deephyper_result_row_to_dict


import requests
import jmespath
from json import JSONDecodeError

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

from lcdb.db.callbacks._base import LCDBCallback



class PCloudRepository(Repository):

    def __init__(self, repo_code, token=None):
        super().__init__()
        self.repo_code = repo_code
        self.content = None
        self.token = token
        # update content
        self.update_content()
        self.root_folder_id = self.content['metadata'].get('folderid')
        self.logger = logging.getLogger(__name__)

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

    def download_result_file_and_get_handle(self, file):

        # get download link
        response = requests.get(f"https://eapi.pcloud.com/getpublinkdownload?code={self.repo_code}&fileid={file}").json()
        download_link = "https://" + response["hosts"][0] + response["path"]

        # download file
        t_start = time.time()
        self.logger.info(f"Starting download of {download_link}")
        response = requests.get(download_link)
        t_end_dl = time.time()

        size_bytes = int(response.headers.get("Content-Length", 0))
        size_mb = int(size_bytes / (1024 * 1024))
        self.logger.info(f"Download of file with size {size_mb}MB finished after {int(1000 * (t_end_dl - t_start))} miliseconds")
        if response.status_code == 200:

            t_start = time.time()
            if download_link.endswith((".gz", ".gzip")):
                filehandle = io.BytesIO(response.content)
            else:
                filehandle = download_link
            return filehandle
        else:
            self.logger.error(f"Failed to fetch the file. Status code: {response.status_code}")

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
        # print(f"Updated content: {self.content}")  # Debugging

        parts = full_path.split("/")
        parent_folder_id = self.root_folder_id
        # print(f"Root folder ID: {parent_folder_id}")  # Debugging

        for part in parts:
            # print(f"Checking folder: '{part}' in parent {parent_folder_id}")  # Debugging

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

    def add_results(self, campaign, *result_files, logs_included=False):
        """
        Uploads result files (JSONL or CSV) to pCloud, preserving lcdb/data/<workflow>/<campaign>/<openmlid>/<file> structure.
        Always uploads logs_<openmlid>.zip if logs_included=True, even if the result file is missing or failed to upload.
        """
        import os, re, io, gzip, json, tempfile, zipfile, pandas as pd, requests

        self.update_content()

        for result_file in result_files:
            result_exists = os.path.exists(result_file)
            is_jsonl = result_file.endswith((".jsonl", ".jsonl.gz", ".jsonl.gzip"))

            # ------------------------------------------------------------
            # Extract workflow and openmlid from path (works even if file missing)
            # ------------------------------------------------------------
            workflow = "unknown_workflow"
            openmlid = "unknown_dataset"
            parts = result_file.split("/")

            for part in parts:
                if part.startswith("lcdb.workflow"):
                    workflow = part.split("-")[0]
                if re.fullmatch(r"\d+", part):
                    openmlid = part

            # ------------------------------------------------------------
            # Derive seeds (only if result file exists)
            # ------------------------------------------------------------
            workflow_seed = valid_seed = test_seed = 0
            if result_exists and is_jsonl:
                try:
                    opener = gzip.open if result_file.endswith((".gz", ".gzip")) else open
                    with opener(result_file, "rt", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                        if first_line:
                            record = json.loads(first_line)
                            workflow_seed = int(record.get("workflow_seed", 0))
                            valid_seed = int(record.get("valid_seed", 0))
                            test_seed = int(record.get("test_seed", 0))
                except Exception as e:
                    print(f"Warning: Could not extract seeds from {result_file}: {e}")

            # ------------------------------------------------------------
            # Build pCloud folder path
            # ------------------------------------------------------------
            path = f"data/{workflow}/{campaign}/{openmlid}"
            folder_id = self._get_or_create_folder_id(path)
            base_name = f"{workflow_seed}-{test_seed}-{valid_seed}.{'jsonl.gz' if is_jsonl else 'csv.gz'}"

            # ------------------------------------------------------------
            # Upload result file (isolated from logs)
            # ------------------------------------------------------------
            try:
                if result_exists:
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
                    else:
                        # compress CSV
                        if result_file.endswith((".gz", ".gzip")):
                            with gzip.open(result_file, "rt", encoding="utf-8") as f:
                                df = pd.read_csv(f)
                        else:
                            df = pd.read_csv(result_file)
                        csv_buf = io.BytesIO()
                        with gzip.GzipFile(fileobj=csv_buf, mode="wb") as gz:
                            gz.write(df.to_csv(index=False).encode("utf-8"))
                        csv_buf.seek(0)
                        upload_buf = csv_buf

                    url = (
                        f"https://eapi.pcloud.com/uploadfile"
                        f"?code={self.repo_code}&auth={self.token}"
                        f"&folderid={folder_id}&filename={base_name}"
                    )
                    resp = requests.post(
                        url, files={"file": (base_name, upload_buf, "application/gzip")}
                    ).json()

                    if resp.get("result") == 0:
                        print(f"Uploaded result file {base_name} to {path}")
                    else:
                        print(f"Upload failed for {base_name}: {resp}")
                else:
                    print(f"Missing result file {result_file}")
            except Exception as e:
                print(f"Error during result upload for {result_file}: {e}")

            # ------------------------------------------------------------
            # Always upload logs.zip if requested (independent of results)
            # ------------------------------------------------------------
            if not logs_included:
                continue

            try:
                abs_path = os.path.abspath(result_file)
                parts = abs_path.split(os.sep)

                # Find LCDB2 root (case-insensitive)
                lcdb_root = None
                for i, part in enumerate(parts):
                    if part.lower() == "lcdb2":
                        lcdb_root = os.sep.join(parts[: i + 1])
                        break
                if not lcdb_root:
                    print(f"Could not locate LCDB2 root in {abs_path}, using current directory as fallback.")
                    lcdb_root = os.getcwd()

                # Determine workflow dir safely
                try:
                    workflow_idx = parts.index("results") + 1
                    workflow_name = parts[workflow_idx]
                except (ValueError, IndexError):
                    workflow_name = workflow

                # Locate logs and results directories
                results_dir = os.path.join(lcdb_root, "results", workflow_name)
                logs_dir = os.path.join(lcdb_root, "logs", workflow_name)

                zip_name = f"logs.zip"
                print(f"Creating {zip_name} for workflow={workflow_name}, openmlid={openmlid}")

                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, zip_name)
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for base_dir in (results_dir, logs_dir):
                            if not os.path.isdir(base_dir):
                                print(f"Skipping missing directory: {base_dir}")
                                continue
                            for root, _, files in os.walk(base_dir):
                                for f in files:
                                    full_path = os.path.join(root, f)
                                    rel_path = os.path.relpath(full_path, lcdb_root)
                                    zipf.write(full_path, rel_path)

                    # Upload logs_<openmlid>.zip to same folder
                    with open(zip_path, "rb") as f:
                        url = (
                            f"https://eapi.pcloud.com/uploadfile"
                            f"?code={self.repo_code}&auth={self.token}"
                            f"&folderid={folder_id}&filename={zip_name}"
                        )
                        status = requests.post(
                            url, files={"file": (zip_name, f, "application/zip")}
                        ).json()

                    if status.get("result") == 0:
                        print(f"Uploaded {zip_name} for {workflow_name}/{openmlid} to {path}")
                    else:
                        print(f"Failed to upload {zip_name}: {status}")
            except Exception as e:
                print(f"Error uploading logs for {openmlid}: {e}")


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
            validation_seeds=None,
            inclusion_predicate=None
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
            if not filename.endswith(".jsonl.gz"):
                continue
            offset = 9 # remove .jsonl.gz
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
                
            # check inclusion predicate
            if inclusion_predicate is not None and not inclusion_predicate(
                workflow=workflow,
                campaign=campaign,
                openmlid=openmlid,
                workflow_seed=_workflow_seed,
                test_seed=_test_seed,
                val_seed=_val_seed
                ):
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
            validation_seeds=None,
            inclusion_predicate=None
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
                validation_seeds=validation_seeds,
                inclusion_predicate=inclusion_predicate
            )
            result_files = result_files_new if result_files is None else pd.concat([result_files, result_files_new], ignore_index=True)
        return result_files

    def get_result_files_of_workflow(
            self,
            workflow=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            inclusion_predicate=None
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
                validation_seeds=validation_seeds,
                inclusion_predicate=inclusion_predicate
            )
            result_files = result_files_new if result_files is None else pd.concat([result_files, result_files_new], ignore_index=True)
        return result_files

    def get_result_files(
            self,
            workflows=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            inclusion_predicate=None
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
                validation_seeds=validation_seeds,
                inclusion_predicate=inclusion_predicate
            )
            result_files = result_files_new if result_files is None else pd.concat([result_files, result_files_new], ignore_index=True)
        return result_files
    
    def get_count_table(self, **kwargs):

        def count_lines_gzip(path, chunk_size=1024 * 1024):
            count = 0
            with gzip.open(path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    count += chunk.count(b"\n")
            return count

        result_file_df = self.get_result_files(**kwargs)
        cnts = []
        
        pbar = tqdm(total=len(result_file_df))
        for _, row in result_file_df.iterrows():
            try:
                filehandle = self.download_result_file_and_get_handle(row["fileid"])
                cnt = count_lines_gzip(filehandle)
            except Exception as e:
                cnt = 0
            
            cnts.append(cnt)
            pbar.update(1)
        result_file_df["num_configs"] = cnts
        pbar.close()
        return result_file_df[["workflow", "campaign", "openmlid", "seed_test", "seed_val", "seed_workflow", "num_configs"]].copy()

    def query_results_as_stream(
            self,
            workflows=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None,
            raise_errors=False,
            report_errors=True,
            inclusion_predicate=None,
            callbacks=None,
            batch_size=100,
            buffer_size=50,
            file_download_buffer=1
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

        # check that all callbacks are proper
        if callbacks is None:
            callbacks = []
        for cb in callbacks:
            if not isinstance(cb, LCDBCallback):
                raise ValueError(f"Expected callback of type LCDBCallback but got {type(cb)}")

        # get all result files
        result_files = self.get_result_files(
            workflows=workflows,
            campaigns=campaigns,
            openmlids=openmlids,
            workflow_seeds=workflow_seeds,
            test_seeds=test_seeds,
            validation_seeds=validation_seeds,
            inclusion_predicate=inclusion_predicate
        )
        result_files["generated_rows"] = 0
        result_files["delivered_rows"] = 0
        result_files["generated_all"] = False
        result_files["delivered_all"] = False
        
        print(result_files)

        # read in all result files
        def gen_fun(
            raise_errors=False,
            max_workers=4
            ):

            total_entries = 0
            if result_files is None:
                return 

            result_row_queue = Queue(maxsize=buffer_size) # maximum number of result rows that can be stored in memory at once, to prevent memory overflow. Adjust as needed.

            # This worker runs in threadpool
            def worker(idx, copy_of_record):
                try:
                    filehandle = self.download_result_file_and_get_handle(copy_of_record["fileid"])
                    with gzip.open(filehandle, 'rt', encoding='utf-8') as f:
                        reader = jsonlines.Reader(f)
                        for row in reader:
                            result_row_queue.put((idx, convert_deephyper_result_row_to_dict(row)))
                            result_files.loc[idx, "generated_rows"] += 1
                    result_files.loc[idx, "generated_all"] = True

                except Exception as e:
                    is_parsing_error = isinstance(e, JSONDecodeError)
                    if is_parsing_error:
                        error_msg = f"Parsing error {repr(e)} for result file:"
                    else:
                        error_msg = f"{type(e)} with message '{repr(e)}' in result file:"

                    error_msg += ""\
                                f"\n\tworkflow {copy_of_record['workflow']}"\
                                f"\n\tcampaign {copy_of_record['campaign']}"\
                                f"\n\topenmlid {copy_of_record['openmlid']}"\
                                f"\n\tseed_wf {copy_of_record['seed_workflow']}"\
                                f"\n\tseed_test {copy_of_record['seed_test']}"\
                                f"\n\tseed_valid {copy_of_record['seed_val']}"\
                                f"\n\tpCloud file id {copy_of_record['fileid']}"
                    if raise_errors:
                        raise Exception(error_msg)
                    elif report_errors:
                        print(error_msg)

            # Producer starts the workers
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(worker, idx, copy_of_record) for idx, copy_of_record in result_files.iterrows()]

                # Meanwhile, yield from queue until all workers finish
                finished = 0
                total = len(futures)
                
                indices_and_rows = []
                while finished < total or not result_row_queue.empty():
                    try:
                        while len(indices_and_rows) < batch_size:
                            indices_and_rows.append(result_row_queue.get(timeout=5))
                        yield [row for _, row in indices_and_rows]
                        
                        # update delivered rows count
                        for idx, raw_row in indices_and_rows:
                            result_files.loc[idx, "delivered_rows"] += 1
                            result_frame_row = result_files.loc[idx]
                            
                            # check callbacks
                            if result_frame_row["generated_all"] and (result_frame_row["delivered_rows"] == result_frame_row["generated_rows"]):
                                result_files.loc[idx, "delivered_all"] = True

                                if callbacks and all(result_files.loc[result_files["openmlid"] == raw_row["openmlid"], "delivered_all"]):
                                    print(f"Running {len(callbacks)} finish callbacks for openmlid {raw_row['openmlid']} for which {result_frame_row['generated_rows']} rows were generated and {result_frame_row['delivered_rows']} were delivered")
                                    for cb in callbacks:
                                        print(f"Running {cb}")
                                        cb.on_workflow_dataset_combination_finished(
                                            workflow=raw_row["workflow"],
                                            openmlid=raw_row["openmlid"],
                                            total_num_records=result_frame_row["delivered_rows"]
                                        )

                        # reset
                        indices_and_rows = []

                    except:
                        pass

                    # Check if any workers have finished
                    finished_now = sum(f.done() for f in futures)
                    if finished_now > finished:
                        self.logger.info(f"{finished_now}/{total} files processed.")
                        finished = finished_now

        return gen_fun()

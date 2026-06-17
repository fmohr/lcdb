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
import urllib.parse

import pandas as pd
from tqdm import tqdm
import requests
import jmespath
from json import JSONDecodeError

from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from lcdb.db._repository import Repository
from lcdb.builder.utils import convert_deephyper_result_row_to_dict
from lcdb.db.callbacks._base import LCDBCallback


class BasePCloudRepository:
    def __init__(self, repo_code=None, token=None, api_host="eapi.pcloud.com"):
        self.repo_code = repo_code
        self.content = None
        self.token = token
        self.api_host = api_host
        if self.repo_code:
            self.update_content()
            self.root_folder_id = self.content['metadata'].get('folderid')
        else:
            self.root_folder_id = 0

    def _load_env(self):
        """Loads environmental variables from the .env file in the workspace."""
        env = {}
        for path in [".env", "../.env", "../../.env", "../../../.env", "../../../../.env"]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            env[key.strip()] = val.strip()
                break
        return env

    def _save_token_to_env(self, token):
        """Saves the access token to the .env file in the workspace."""
        env_path = None
        for path in [".env", "../.env", "../../.env", "../../../.env", "../../../../.env"]:
            if os.path.exists(path):
                env_path = path
                break
        if not env_path:
            env_path = ".env"
        
        lines = []
        token_written = False
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("ACCESS_TOKEN="):
                        lines.append(f"ACCESS_TOKEN={token}\n")
                        token_written = True
                    else:
                        lines.append(line)
        
        if not token_written:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"ACCESS_TOKEN={token}\n")
            
        with open(env_path, "w") as f:
            f.writelines(lines)
        print(f"Saved ACCESS_TOKEN to {env_path} file for future use.")

    def update_content(self):
        """Fetches repository content metadata from pCloud."""
        if self.repo_code:
            self.content = requests.get(f"https://{self.api_host}/showpublink?code={self.repo_code}").json()

    def authenticate(self, client_id=None, client_secret=None, redirect_uri=None):
        """Authenticates using OAuth 2.0 flow."""
        env = self._load_env()
        
        # Check if ACCESS_TOKEN is already provided in .env or environment
        access_token = env.get("ACCESS_TOKEN") or os.environ.get("ACCESS_TOKEN")
        host_name = env.get("HOST_NAME") or os.environ.get("HOST_NAME")
        
        if access_token:
            self.token = access_token
            if host_name:
                self.api_host = host_name
            print("Found ACCESS_TOKEN in configuration. Using cached token.")
            return

        client_id = client_id or env.get("CLIENT_ID") or os.environ.get("CLIENT_ID")
        client_secret = client_secret or env.get("CLIENT_SECRET") or os.environ.get("CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError(
                "OAuth 2.0 authentication requires 'CLIENT_ID' and 'CLIENT_SECRET'. "
                "Please ensure they are defined in your .env file or passed directly."
            )
        
        # Check if ACCESS_CODE is already provided in .env or environment
        access_code = env.get("ACCESS_CODE") or os.environ.get("ACCESS_CODE")
        
        if access_code:
            print("Found ACCESS_CODE in configuration. Skipping interactive prompt...")
            code = access_code
            if host_name:
                self.api_host = host_name
                print(f"Using configured API host: {self.api_host}")
            hostname = self.api_host
        else:
            # Step 1: Generate Authorization URL
            auth_url = f"https://my.pcloud.com/oauth2/authorize?client_id={client_id}&response_type=code"
            if redirect_uri:
                auth_url += f"&redirect_uri={urllib.parse.quote(redirect_uri)}"
            
            print("\n" + "="*80)
            print("1. Please open the following URL in your web browser to authorize the app:")
            print(auth_url)
            print("="*80 + "\n")
            
            # Step 2: Prompt user for code or redirect URL
            user_input = input("2. Enter the authorization code (or the full redirect URL if redirected): ").strip()
            
            code = user_input
            hostname = self.api_host
            
            # Parse code and hostname if user pasted the entire redirect URL
            if "code=" in user_input:
                parsed = urllib.parse.urlparse(user_input)
                query_params = urllib.parse.parse_qs(parsed.query)
                if "code" in query_params:
                    code = query_params["code"][0]
                if "hostname" in query_params:
                    hostname = query_params["hostname"][0]
                    self.api_host = hostname
                    print(f"Detected regional API host: {hostname}")
        
        # Step 3: Exchange code for access token
        token_url = f"https://{hostname}/oauth2_token"
        params = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code
        }
        
        response = requests.post(token_url, data=params).json()
        if "access_token" in response:
            self.token = response["access_token"]
            print("Authentication successful! Token retrieved and stored.")
            # Persist the token to .env
            self._save_token_to_env(self.token)
        else:
            raise ValueError(f"Failed to obtain access token: {response}")

    def _get_api_params(self, **kwargs):
        """Constructs query parameters for pCloud API requests."""
        params = {}
        if self.token:
            params["access_token"] = self.token
        if self.repo_code:
            params["code"] = self.repo_code
        for k, v in kwargs.items():
            if v is not None:
                params[k] = v
        return params

    def _create_folder(self, parent_folder_id, name):
        """Creates a new folder in pCloud."""
        url = f"https://{self.api_host}/createfolder"
        params = self._get_api_params(folderid=parent_folder_id, name=name)
        response = requests.get(url, params=params).json()
        if response.get("result") != 0:
            raise ValueError(f"Failed to create folder '{name}': {response}")
        self.update_content()
        return response["metadata"]["folderid"]

    def _get_or_create_folder_id(self, full_path):
        """Gets or creates the folder ID for the specified full path."""
        if not full_path or full_path == "/" or full_path == "":
            return self.root_folder_id

        parts = full_path.strip("/").split('/')
        parent_folder_id = self.root_folder_id

        for part in parts:
            url = f"https://{self.api_host}/listfolder"
            params = self._get_api_params(folderid=parent_folder_id)
            response = requests.get(url, params=params).json()
            
            contents = response.get("metadata", {}).get("contents", [])
            folder = next((item for item in contents if item["name"] == part and item["isfolder"]), None)
            if folder:
                parent_folder_id = folder["folderid"]
            else:
                parent_folder_id = self._create_folder(parent_folder_id, part)

        return parent_folder_id

    def get_download_link(self, file_id):
        """Gets a direct download link for a file by its file ID.
        Uses getpublinkdownload if repo_code is set, otherwise getfilelink."""
        if self.repo_code:
            url = f"https://{self.api_host}/getpublinkdownload"
            params = self._get_api_params(fileid=file_id)
        else:
            if not self.token:
                raise ValueError("Authentication token is required to get a file link.")
            url = f"https://{self.api_host}/getfilelink"
            params = self._get_api_params(fileid=file_id)
            
        response = requests.get(url, params=params).json()
        if response.get("result") != 0:
            raise RuntimeError(f"Failed to get download link for file ID {file_id}: {response.get('error', 'Unknown error')}")
            
        host = response["hosts"][0]
        path = response["path"]
        return f"https://{host}{path}"

    def download_file_by_id(self, file_id, local_dest_path, filename=None):
        """Downloads a file by its pCloud file ID to a local path."""
        download_url = self.get_download_link(file_id)
        
        if os.path.isdir(local_dest_path) or local_dest_path.endswith("/") or local_dest_path.endswith("\\"):
            target_name = filename or os.path.basename(urllib.parse.urlparse(download_url).path)
            local_file_dest = os.path.join(local_dest_path, target_name)
        else:
            local_file_dest = local_dest_path
            
        os.makedirs(os.path.dirname(os.path.abspath(local_file_dest)), exist_ok=True)
        
        r = requests.get(download_url, stream=True)
        r.raise_for_status()
        with open(local_file_dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return local_file_dest



class PCloudRepository(BasePCloudRepository, Repository):

    def __init__(self, repo_code, token=None):
        BasePCloudRepository.__init__(self, repo_code=repo_code, token=token)
        Repository.__init__(self)
        self.logger = logging.getLogger(__name__)

    def exists(self):
        return self.content is not None and len(self.content) > 0

    def authenticate(self, username=None, password=None, device="lcdbclient", authexpire=300, client_id=None, client_secret=None, redirect_uri=None):
        """
        Supports both legacy password-based authentication and OAuth 2.0.
        """
        if username is not None and password is not None:
            url = f"https://{self.api_host}/userinfo?getauth=1&logout=1&device={device}&authexpire={authexpire}"
            response = requests.post(url, {
                "username": username,
                "password": password
            }).json()
            self.token = response["auth"] if "auth" in response else None
            if self.token is None:
                raise ValueError(f"Authentication failed. Response from server was {response}.")
        else:
            # Fall back to OAuth 2.0 authentication
            BasePCloudRepository.authenticate(self, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)


    def download_result_file_and_get_handle(self, file):
        # get download link using generalized base class method
        download_link = self.get_download_link(file)

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
            
            # Using safe base helper to get listfolder parameters
            url = f"https://{self.api_host}/listfolder"
            params = self._get_api_params(folderid=folder_id)
            self.content = requests.get(url, params=params).json()  # Update content to current folder

        return folder_id

    @staticmethod
    def _extract_result_metadata(result_file):
        """Infer workflow, dataset, and seeds from the LCDB result path."""
        normalized_path = os.path.normpath(result_file)
        path_for_matching = normalized_path.replace("\\", "/")
        parts = [part for part in re.split(r"[\\/]+", normalized_path) if part]

<<<<<<< HEAD
    def _get_or_create_folder_id(self, full_path):
        """Gets or creates the folder ID for the specified path."""
        self.update_content()
        # print(f"Updated content: {self.content}")  # Debugging
=======
        workflow = "unknown_workflow"
        openmlid = "unknown_dataset"
        workflow_seed = 0
        test_seed = 0
        valid_seed = 0
>>>>>>> 18e1d0ed7568dc47c24d819032b2030d9178173d

        if "/results/" in path_for_matching:
            lcdb_root = os.path.normpath(path_for_matching.split("/results/", 1)[0] or os.sep)
        else:
            lcdb_root = os.getcwd()

        for idx, part in enumerate(parts):
            if part == "results" and idx + 1 < len(parts):
                for workflow_idx in range(idx + 1, len(parts)):
                    if not parts[workflow_idx].startswith("lcdb.workflow"):
                        continue

                    workflow = parts[workflow_idx]

                    if workflow_idx + 1 < len(parts) and re.fullmatch(r"\d+", parts[workflow_idx + 1]):
                        openmlid = parts[workflow_idx + 1]

                    if workflow_idx + 2 < len(parts):
                        seed_match = re.fullmatch(
                            r"(?P<valid>\d+)-(?P<test>\d+)-(?P<workflow>\d+)",
                            parts[workflow_idx + 2],
                        )
                        if seed_match:
                            valid_seed = int(seed_match.group("valid"))
                            test_seed = int(seed_match.group("test"))
                            workflow_seed = int(seed_match.group("workflow"))
                    break
                break

            if part.startswith("lcdb.workflow"):
                workflow = part.split("-")[0]
            elif re.fullmatch(r"\d+", part):
                openmlid = part

        return {
            "lcdb_root": lcdb_root,
            "workflow": workflow,
            "openmlid": openmlid,
            "workflow_seed": workflow_seed,
            "test_seed": test_seed,
            "valid_seed": valid_seed,
        }

    @staticmethod
    def _read_seeds_from_result_file(result_file):
        opener = gzip.open if result_file.endswith((".gz", ".gzip")) else open
        with opener(result_file, "rt", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            record = json.loads(first_line)
        return {
            "workflow_seed": int(record.get("workflow_seed", 0)),
            "valid_seed": int(record.get("valid_seed", 0)),
            "test_seed": int(record.get("test_seed", 0)),
        }

    @staticmethod
    def _get_relevant_log_files(lcdb_root, workflow, openmlid):
        workflow_log_root = os.path.join(lcdb_root, "logs", workflow)
        prefix = f"o:{openmlid}-"
        matches = []
        for subdir, suffix in (("out", ".log"), ("err", ".err")):
            directory = os.path.join(workflow_log_root, subdir)
            if not os.path.isdir(directory):
                continue
            for filename in os.listdir(directory):
                if filename.startswith(prefix) and filename.endswith(suffix):
                    matches.append(os.path.join(directory, filename))

        return sorted(matches)

<<<<<<< HEAD
        # If all parts are found, return the final parent_folder_id
        return parent_folder_id

    @staticmethod
    def _extract_result_metadata(result_file):
        """Infer workflow, dataset and seeds from the LCDB result path."""
        normalized_path = os.path.normpath(result_file)
        path_for_matching = normalized_path.replace("\\", "/")
        parts = [part for part in re.split(r"[\\/]+", normalized_path) if part]

        workflow = "unknown_workflow"
        openmlid = "unknown_dataset"
        workflow_seed = 0
        test_seed = 0
        valid_seed = 0

        if "/results/" in path_for_matching:
            lcdb_root = os.path.normpath(path_for_matching.split("/results/", 1)[0] or os.sep)
        else:
            lcdb_root = os.getcwd()

        if "/results/" in path_for_matching:
            path_inside_results = path_for_matching.split("/results/", 1)[1]
            results_parts = [part for part in path_inside_results.split("/") if part]

            workflow_index = None
            if len(results_parts) >= 1 and results_parts[0].startswith("lcdb.workflow"):
                workflow_index = 0
            elif len(results_parts) >= 2 and results_parts[1].startswith("lcdb.workflow"):
                workflow_index = 1

            if workflow_index is not None:
                workflow = results_parts[workflow_index]

                if len(results_parts) > workflow_index + 1 and re.fullmatch(r"\d+", results_parts[workflow_index + 1]):
                    openmlid = results_parts[workflow_index + 1]

                if len(results_parts) > workflow_index + 2:
                    seed_match = re.fullmatch(r"(?P<valid>\d+)-(?P<test>\d+)-(?P<workflow>\d+)", results_parts[workflow_index + 2])
                    if seed_match:
                        valid_seed = int(seed_match.group("valid"))
                        test_seed = int(seed_match.group("test"))
                        workflow_seed = int(seed_match.group("workflow"))

                return {
                    "lcdb_root": lcdb_root,
                    "workflow": workflow,
                    "openmlid": openmlid,
                    "workflow_seed": workflow_seed,
                    "test_seed": test_seed,
                    "valid_seed": valid_seed,
                }

        for part in parts:
            if part.startswith("lcdb.workflow"):
                workflow = part.split("-")[0]
            elif re.fullmatch(r"\d+", part):
                openmlid = part

        return {
            "lcdb_root": lcdb_root,
            "workflow": workflow,
            "openmlid": openmlid,
            "workflow_seed": workflow_seed,
            "test_seed": test_seed,
            "valid_seed": valid_seed,
        }

    @staticmethod
    def _read_seeds_from_result_file(result_file):
        opener = gzip.open if result_file.endswith((".gz", ".gzip")) else open
        with opener(result_file, "rt", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            record = json.loads(first_line)
        return {
            "workflow_seed": int(record.get("workflow_seed", 0)),
            "valid_seed": int(record.get("valid_seed", 0)),
            "test_seed": int(record.get("test_seed", 0)),
        }

    @staticmethod
    def _get_relevant_log_files(lcdb_root, workflow, openmlid):
        workflow_log_root = os.path.join(lcdb_root, "logs", workflow)
        prefix = f"o:{openmlid}-"
        matches = []
        for subdir, suffix in (("out", ".log"), ("err", ".err")):
            directory = os.path.join(workflow_log_root, subdir)
            if not os.path.isdir(directory):
                continue
            for filename in os.listdir(directory):
                if filename.startswith(prefix) and filename.endswith(suffix):
                    matches.append(os.path.join(directory, filename))

        return sorted(matches)

=======
>>>>>>> 18e1d0ed7568dc47c24d819032b2030d9178173d
    def add_results(self, campaign, *result_files, logs_included=False):
        """
        Uploads result files (JSONL or CSV) to pCloud, preserving lcdb/data/<workflow>/<campaign>/<openmlid>/<file> structure.
        If requested, uploads a per-run log archive with only the log files that match the
        dataset and seed combination inferred from the result path.
        """
        import io, tempfile

        self.update_content()

        for result_file in result_files:
            result_exists = os.path.exists(result_file)
            is_jsonl = result_file.endswith((".jsonl", ".jsonl.gz", ".jsonl.gzip"))

            metadata = self._extract_result_metadata(result_file)
            workflow = metadata["workflow"]
            openmlid = metadata["openmlid"]
            workflow_seed = metadata["workflow_seed"]
            test_seed = metadata["test_seed"]
            valid_seed = metadata["valid_seed"]

            if result_exists and is_jsonl:
                try:
                    result_seeds = self._read_seeds_from_result_file(result_file)
                    if result_seeds is not None:
                        workflow_seed = result_seeds["workflow_seed"]
                        valid_seed = result_seeds["valid_seed"]
                        test_seed = result_seeds["test_seed"]
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

                    # Using safe base parameter builder
                    url = f"https://{self.api_host}/uploadfile"
                    params = self._get_api_params(folderid=folder_id, filename=base_name)
                    resp = requests.post(
                        url, params=params, files={"file": (base_name, upload_buf, "application/gzip")}
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
<<<<<<< HEAD

            try:
                log_files = self._get_relevant_log_files(
                    metadata["lcdb_root"],
                    workflow,
                    openmlid,
                )
                if not log_files:
                    print(
                        f"No matching log files found for workflow={workflow}, "
                        f"openmlid={openmlid}"
                    )
                    continue

                zip_name = "logs.zip"
                print(f"Creating {zip_name} for workflow={workflow}, openmlid={openmlid}")

                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, zip_name)
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for log_file in log_files:
                            rel_path = os.path.relpath(log_file, metadata["lcdb_root"])
                            zipf.write(log_file, rel_path)

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
                        print(f"Uploaded {zip_name} for {workflow}/{openmlid} to {path}")
                    else:
                        print(f"Failed to upload {zip_name}: {status}")
            except Exception as e:
                print(f"Error uploading logs for {openmlid}: {e}")
=======
>>>>>>> 18e1d0ed7568dc47c24d819032b2030d9178173d

            try:
                log_files = self._get_relevant_log_files(
                    metadata["lcdb_root"],
                    workflow,
                    openmlid,
                )
                if not log_files:
                    print(
                        f"No matching log files found for workflow={workflow}, "
                        f"openmlid={openmlid}"
                    )
                    continue

                zip_name = "logs.zip"
                print(f"Creating {zip_name} for workflow={workflow}, openmlid={openmlid}")

                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, zip_name)
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for log_file in log_files:
                            rel_path = os.path.relpath(log_file, metadata["lcdb_root"])
                            zipf.write(log_file, rel_path)

                    with open(zip_path, "rb") as f:
                        # Using safe base parameter builder
                        url = f"https://{self.api_host}/uploadfile"
                        params = self._get_api_params(folderid=folder_id, filename=zip_name)
                        status = requests.post(
                            url, params=params, files={"file": (zip_name, f, "application/zip")}
                        ).json()

                    if status.get("result") == 0:
                        print(f"Uploaded {zip_name} for {workflow}/{openmlid} to {path}")
                    else:
                        print(f"Failed to upload {zip_name}: {status}")
            except Exception as e:
                print(f"Error uploading logs for {openmlid}: {e}")

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
        
        # get datasets
        datasets = []
        for i in qry_result:
            try:
                openmlid = int(i)
                datasets.append(openmlid)
            except ValueError:
                pass
        return sorted(datasets)

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
                            self.logger.debug("Adding result row")
                            result_row_queue.put((idx, convert_deephyper_result_row_to_dict(row)))
                            result_files.loc[idx, "generated_rows"] += 1
                            self.logger.debug(f'Number of generated rows is now {result_files.loc[idx, "generated_rows"]}')
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
                    self.logger.info("Waiting for results")
                    try:
                        while len(indices_and_rows) < batch_size:
                            indices_and_rows.append(result_row_queue.get(timeout=5))
                    except:
                        pass
                    self.logger.info(f"Filled up a batch of size {len(indices_and_rows)}, returning it.")
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

                    # Check if any workers have finished
                    finished_now = sum(f.done() for f in futures)
                    if finished_now > finished:
                        self.logger.info(f"{finished_now}/{total} files processed.")
                        finished = finished_now

                self.logger.info("Leaving")
        return gen_fun()

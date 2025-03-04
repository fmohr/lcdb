from curses import meta
import os
import requests
import jmespath
import tempfile
import json

class PCloudRepository:
    def __init__(self, repo_code, token=None):
        self.repo_code = repo_code
        self.content = None
        self.token = token
        self.update_content()
        self.root_folder_id = self.content['metadata'].get('folderid')

    def update_content(self):
        """Fetches repository content metadata from pCloud."""
        self.content = requests.get(f"https://eapi.pcloud.com/showpublink?code={self.repo_code}").json()

    def exists(self):
        """Checks if the repository content is available."""
        return bool(self.content)

    def authenticate(self, username, password, device="slmclient", authexpire=31536000):
        """Authenticates the user to receive an authorization token."""
        url = f"https://eapi.pcloud.com/userinfo?getauth=1&logout=1&device={device}&authexpire={authexpire}"
        response = requests.post(url, {"username": username, "password": password}).json()
        self.token = response.get("auth")
        if self.token is None:
            raise ValueError(f"Authentication failed. Response: {response}.")

    def _get_folder_id(self, path=None, root=False):
        """Returns the folder ID for a specified full path within the repository."""
        if root:
            return self.root_folder_id
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
        """Creates a new folder in pCloud."""
        response = requests.get(
            f"https://eapi.pcloud.com/createfolder?code={self.repo_code}&auth={self.token}&folderid={parent_folder_id}&name={name}"
        ).json()
        if response.get("result") != 0:
            raise ValueError(f"Failed to create folder '{name}': {response}")
        self.update_content()
        return response["metadata"]["folderid"]

    def _get_or_create_folder_id(self, full_path):
        """Gets or creates the folder ID for the specified full path."""
        parts = full_path.split('/')
        parent_folder_id = self.root_folder_id

        for part in parts:
            response = requests.get(
                f"https://eapi.pcloud.com/listfolder?code={self.repo_code}&auth={self.token}&folderid={parent_folder_id}"
            ).json()
            folder = next((item for item in response.get("metadata", {}).get("contents", [])
                           if item["name"] == part and item["isfolder"]), None)

            if folder:
                parent_folder_id = folder["folderid"]
            else:
                parent_folder_id = self._create_folder(parent_folder_id, part)

        return parent_folder_id
    
    def sync_local_to_pcloud(self, local_path, parent_folder_id=None):
        """Synchronizes a local directory structure to pCloud recursively."""
        parent_folder_id = parent_folder_id or self.root_folder_id
        for item in os.listdir(local_path):
            item_path = os.path.join(local_path, item)
            if os.path.isdir(item_path):
                folder_id = self._get_or_create_folder_id(parent_folder_id, item)
                self.sync_local_to_pcloud(item_path, folder_id)
            else:
                self._upload_file(parent_folder_id, item_path, item)

    def add_results(self, results, file_name):
        """Uploads a CSV file containing training results to pCloud."""
        self.update_content()
        folder_id = self._get_folder_id(root=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, file_name)
            results.to_csv(csv_path, index=False)
            self._upload_file(folder_id, csv_path, file_name)

    def upload_model(self, metadata):
        """
        Uploads a model from a local path to pCloud.
        
        Args:
            metadata: dict containing:
                - load_path: Path to load model from locally
                - save_path: Path to save model to in pCloud
        """
        self.update_content()
        
        # Get or create folder for save path
        folder_id = self._get_folder_id(metadata["save_path"]) or self._get_or_create_folder_id(metadata["save_path"])
        
        print(f"Uploading model to folder id: {folder_id}")
        
        # Upload all files from load path
        if os.path.exists(metadata["load_path"]):
            # Upload all files in directory
            for item in os.listdir(metadata["load_path"]):
                item_path = os.path.join(metadata["load_path"], item)
                if os.path.isfile(item_path):
                    self._upload_file(folder_id, item_path, item)
                elif os.path.isdir(item_path):
                    subfolder_id = self._get_or_create_folder_id(os.path.join(metadata["save_path"], item))
                    self.sync_local_to_pcloud(item_path, subfolder_id)
        else:
            raise ValueError(f"Load path does not exist: {metadata['load_path']}")
        
        
    def add_model(self, metadata):
        """Saves a model and its configuration to pCloud."""
        self.update_content()
        dataset_name = metadata["dataset_name"]
        model_size = metadata["model_size"]
        anchor = metadata["token_anchor"]
        seed = metadata["seed"]
        path = f"output/{dataset_name}/{model_size}/seed-{seed}/{anchor}/pretrained"
        
        folder_id = self._get_folder_id(path) or self._get_or_create_folder_id(path)

        print(f"Folder id: {folder_id}")
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata["trainer"].save_model(temp_dir)
            metadata["tokenizer"].save_pretrained(temp_dir)
            trainer_state = {
                "best_metric": metadata["trainer"].state.best_metric,
                "epoch": metadata["trainer"].state.epoch,
                "seed": seed,
                "global_step": metadata["trainer"].state.global_step,
                "log_history": metadata["trainer"].state.log_history,
                "total_flos": metadata["trainer"].state.total_flos,
            }
            with open(os.path.join(temp_dir, "trainer_state.json"), "w") as f:
                json.dump(trainer_state, f, indent=4)

            logs_folder_id = self._get_or_create_folder_id("logs")
            log_name = f"{model_size}-{dataset_name}-{seed}-{anchor}.log"
            self._upload_file(logs_folder_id, f"logs/{log_name}", log_name)
            self.sync_local_to_pcloud(temp_dir, folder_id)

    def _upload_file(self, folder_id, file_path, file_name):
        """Uploads a file to a specific pCloud folder."""
        url = f"https://eapi.pcloud.com/uploadfile?code={self.repo_code}&auth={self.token}&folderid={folder_id}&filename={file_name}"
        with open(file_path, 'rb') as f:
            response = requests.post(url, files={"file": (file_name, f)})
            response.raise_for_status()
            return response.json()

    def add_json_results(self, results, file_name):
        """Uploads a JSON file containing results to pCloud."""
        self.update_content()
        folder_path = os.path.dirname(file_name)
        folder_id = self._get_folder_id(folder_path) or self._get_or_create_folder_id(folder_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, os.path.basename(file_name))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)  # Write full JSON object

            # Upload JSON file to pCloud
            self._upload_file(folder_id, json_path, os.path.basename(file_name))

    def fetch_blimp_results(self, dataset, model_size, seed, anchor):
        """Fetches and loads a JSON file from pCloud based on dataset, model size, and anchor."""
        # Formulate the path and folder ID
        self.update_content()

        path = f"output/{dataset}/{model_size}/seed-{seed}/{anchor}"
        folder_id = self._get_folder_id(path)
        
        if folder_id is None:
            raise ValueError(f"Folder for path '{path}' not found.")

        # List folder contents and look for the JSON file
        response = requests.get(
            f"https://eapi.pcloud.com/listfolder?code={self.repo_code}&auth={self.token}&folderid={folder_id}"
        ).json()
        
        # Search for the 'blimp_results.json' file in the specified path
        json_file = next((item for item in response.get("metadata", {}).get("contents", [])
                          if item["name"] == "blimp_results.json" and not item["isfolder"]), None)

        if not json_file:
            raise FileNotFoundError(f"'blimp_results.json' not found in path '{path}'.")

        # Get the download link for the JSON file
        file_id = json_file["fileid"]
        download_url_response = requests.get(
            f"https://eapi.pcloud.com/getfilelink?fileid={file_id}&auth={self.token}"
        ).json()

        if "hosts" in download_url_response and "path" in download_url_response:
            host = download_url_response["hosts"][0]
            download_path = download_url_response["path"]
            download_url = f"https://{host}{download_path}"

            # Download the JSON file and load its contents
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_json_file:
                temp_json_file.write(response.content)
                temp_json_path = temp_json_file.name

            # Load the JSON content into memory
            with open(temp_json_path, 'r', encoding="utf-8") as f:
                json_data = json.load(f)
                
            os.remove(temp_json_path)  # Clean up temporary file
            return json_data
        
        raise ValueError("Could not generate a valid download URL for the JSON file.")




    def download_model_recursive(self, path):
        """
        Downloads the specified path directory and all its subdirectories recursively to a persistent temporary folder.
        
        :param path: The full path in the form '<dataset>/<model_size>/<anchor>'.
        :return: Path to the temporary folder containing all downloaded files and subdirectories.
        """
        self.update_content()
        folder_id = self._get_folder_id(path)
        if folder_id is None:
            raise ValueError(f"Folder for path '{path}' not found.")

        temp_dir = tempfile.mkdtemp()

        def download_folder(folder_id, current_path):
            """Helper function to recursively download folders and files"""
            response = requests.get(
                f"https://eapi.pcloud.com/listfolder?code={self.repo_code}&auth={self.token}&folderid={folder_id}"
            ).json()

            if "metadata" in response and "contents" in response["metadata"]:
                for item in response["metadata"]["contents"]:
                    item_name = item["name"]
                    item_path = os.path.join(current_path, item_name)
                    
                    if item["isfolder"]:
                        # Create directory and recurse
                        os.makedirs(item_path, exist_ok=True)
                        download_folder(item["folderid"], item_path)
                    else:
                        # Download file
                        file_id = item["fileid"]
                        download_url_response = requests.get(
                            f"https://eapi.pcloud.com/getfilelink?fileid={file_id}&auth={self.token}"
                        ).json()
                        
                        if "hosts" in download_url_response and "path" in download_url_response:
                            host = download_url_response["hosts"][0]
                            dl_path = download_url_response["path"]
                            download_url = f"https://{host}{dl_path}"

                            response = requests.get(download_url, stream=True)
                            response.raise_for_status()

                            with open(item_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)

        # Start recursive download from root folder
        download_folder(folder_id, temp_dir)
        print(f"Files and folders downloaded recursively to: {temp_dir}")
        return temp_dir
    
    
    def download_model(self, path):
        """
        Downloads the specified path directory to a persistent temporary folder.
        
        :param path: The full path in the form '<dataset>/<model_size>/<anchor>'.

        :return: Path to the temporary folder containing the downloaded files.
        """
        self.update_content()
        folder_id = self._get_folder_id(path)
        if folder_id is None:
            raise ValueError(f"Folder for path '{path}' not found.")

        temp_dir = tempfile.mkdtemp()

        response = requests.get(
            f"https://eapi.pcloud.com/listfolder?code={self.repo_code}&auth={self.token}&folderid={folder_id}"
        ).json()

        if "metadata" in response and "contents" in response["metadata"]:
            for item in response["metadata"]["contents"]:
                if not item["isfolder"]:
                    file_id = item["fileid"]
                    file_name = item["name"]
                    
                    download_url_response = requests.get(
                        f"https://eapi.pcloud.com/getfilelink?fileid={file_id}&auth={self.token}"
                    ).json()
                    
                    if "hosts" in download_url_response and "path" in download_url_response:
                        host = download_url_response["hosts"][0]
                        path = download_url_response["path"]
                        download_url = f"https://{host}{path}"

                        response = requests.get(download_url, stream=True)
                        response.raise_for_status()

                        file_path = os.path.join(temp_dir, file_name)
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

        print(f"Files downloaded to: {temp_dir}")
        return temp_dir
import os
import pandas as pd
import gzip

class TracebackExtractor:
    CACHE_FOLDER = os.path.expanduser("~/.lcdb")

    def __init__(self, repos=None):
        self.repos = repos or ["."]

    def add_repo(self, repo):
        if repo not in self.repos:
            self.repos.append(repo)

    @staticmethod
    def _get_subfolders(folder):
        return [f.name for f in os.scandir(folder) if f.is_dir()]

    @staticmethod
    def _get_result_files_in_folder(folder):
        return [
            os.path.join(folder, f.name) for f in os.scandir(folder)
            if f.is_file() and f.name.endswith((".csv", ".csv.gz"))
        ]

    def _filter_result_files(self, result_files, workflow_seed, test_seed, validation_seed):
        filtered_files = []
        for file in result_files:
            name, ext = os.path.splitext(os.path.basename(file))
            if ext == ".gz":
                name, _ = os.path.splitext(name)  # Remove .csv.gz
            
            try:
                _workflow_seed, _test_seed, _val_seed = map(int, name.split("-"))
            except ValueError:
                continue  # Skip files with incorrect naming

            if (workflow_seed is not None and workflow_seed != _workflow_seed) or \
               (test_seed is not None and test_seed != _test_seed) or \
               (validation_seed is not None and validation_seed != _val_seed):
                continue
            
            filtered_files.append(file)

        return filtered_files

    def _read_tracebacks(self, file):
        open_func = gzip.open if file.endswith((".gz", ".gzip")) else open
        try:
            df = pd.read_csv(open_func(file, "rt") if file.endswith((".gz", ".gzip")) else file)
            return set(df['traceback'].dropna().unique()) if 'traceback' in df.columns else set()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return set()

    def get_results_for_all_configs(self, workflow, openmlid, workflow_seed=None, test_seed=None, validation_seed=None):
        result_files = [
            file
            for repo in self.repos
            for campaign in self._get_subfolders(repo)
            for file in self._get_result_files_in_folder(os.path.join(repo, campaign, workflow, str(openmlid)))
        ]
        result_files = self._filter_result_files(result_files, workflow_seed, test_seed, validation_seed)

        tracebacks = set()
        for file in result_files:
            tracebacks.update(self._read_tracebacks(file))

        return pd.DataFrame(tracebacks, columns=['traceback'])

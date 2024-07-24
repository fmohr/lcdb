import os.path
import pathlib
import json


lcdb_folder = ".lcdb"


def get_database_location():

    # by default we look if `.lcdb` exist in the current directory
    if pathlib.Path(lcdb_folder).exists():
        return lcdb_folder

    # If it does not exist  we fall back to ~/.lcdb
    default_lcdb_folder = f"~/{lcdb_folder}"
    default_lcdb_path = pathlib.Path(os.path.expanduser(default_lcdb_folder))

    # automatically create folder in home directory if it does not exist
    if not default_lcdb_path.exists():
        init_database(path=default_lcdb_path)

    return default_lcdb_path.absolute()


def init_database(path=None, config=None):

    if path is None:
        path = pathlib.Path(lcdb_folder)
    path.mkdir(exist_ok=True, parents=True)
    print(f"Made {path}")

    default_config = {
        "repositories": {
            "home": "~/.lcdb/data",
            "local": ".lcdb/data"
        }
    }
    if config is not None:
        default_config.update(config)
    config = default_config

    with open(f"{path}/config.json", "w") as f:
        json.dump(config, f)


def get_repository_paths(database_location=None):

    if database_location is None:
        database_location = get_database_location()

    with open(f"{database_location}/config.json", "r") as f:
        cfg = json.load(f)
        candidate_folders = {k: os.path.expanduser(p) for k, p in cfg["repositories"].items()}
        return candidate_folders

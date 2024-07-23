import os.path
import pathlib
import json


lcdb_folder = ".lcdb"


def get_database_location():

    # by default we look if `.lcdb` exist in the current directory
    if pathlib.Path(lcdb_folder).exists():
        return lcdb_folder

    # If it does not exist  we fall back to ~/.lcdb
    if pathlib.Path(f"~/{lcdb_folder}").exists():
        return f"~/{lcdb_folder}"

    raise Exception(
        "LCDB has not been properly initialized on this system. There should be an .lcdb directory in your home folder."
    )


def init_database(config=None):
    path = pathlib.Path(lcdb_folder)
    path.mkdir(exist_ok=True)

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
        candidate_folders = {k: p for k, p in candidate_folders.items() if pathlib.Path(p).exists()}
        return candidate_folders


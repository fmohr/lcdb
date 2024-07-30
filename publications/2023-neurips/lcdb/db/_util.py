import os.path
import pathlib


def get_path_to_lcdb_config(default_lcdb_folder="~/.lcdb", lcdb_config_filename=".lcdb_config.json"):

    # by default we look if `.lcdb_config.json` exist in the current directory
    if pathlib.Path(lcdb_config_filename).exists():
        return f"{os.getcwd()}/{lcdb_config_filename}"

    # If it does not exist we fall back to ~/.lcdb
    default_path = pathlib.Path(os.path.expanduser(f"{default_lcdb_folder}/{lcdb_config_filename}"))
    return default_path.absolute()


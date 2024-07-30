import os.path
import pathlib


def get_path_to_lcdb(lcdb_folder_name=".lcdb"):

    # by default we look if `.lcdb_config.json` exist in the current directory
    if pathlib.Path(lcdb_folder_name).exists():
        return f"{os.getcwd()}/{lcdb_folder_name}"

    # If it does not exist we fall back to ~/.lcdb
    default_path = pathlib.Path(os.path.expanduser(f"~/{lcdb_folder_name}"))
    return default_path.absolute()


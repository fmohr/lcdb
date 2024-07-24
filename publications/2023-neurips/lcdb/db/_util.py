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
        from ._database import LCDB
        LCDB().create(path=default_lcdb_path)

    return default_lcdb_path.absolute()


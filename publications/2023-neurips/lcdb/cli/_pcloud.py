import os
from dotenv import set_key, load_dotenv
from lcdb.db import PCloudRepository
import os

# Avoid Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "pcloud"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="To reload PCloud token."
    )

    subparser.add_argument(
        "--env_path",
        '-p',
        type=str,
        help="Env file path.",
        required=True,
    )

    subparser.set_defaults(func=function_to_call)

def main(**kwargs):
  dotenv_path = kwargs.pop("env_path")
  load_dotenv(dotenv_path)
  # Load the environment variables
  repo_code = os.getenv("PCLOUD_CODE")
  pcloud_username = os.getenv("PCLOUD_USERNAME")
  pcloud_password = os.getenv("PCLOUD_PASSWORD")
  
  # Authenticate with the PCloudRepository
  try:
    repo = PCloudRepository(repo_code=repo_code)
    output = repo.authenticate(username=pcloud_username, password=pcloud_password, authexpire=86400*2)
    set_key(dotenv_path, "PCLOUD_TOKEN", repo.token)
  except ValueError as e:
      if "Authentication failed" in str(e):
          print(f"Error: {e}")
          exit(1)  # Exit the script if authentication fails

  # Set the token in the .env file

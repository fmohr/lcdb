import os
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
  
  # lazy import so this is only required if one really uses the functionality
  from dotenv import set_key, load_dotenv

  dotenv_path = kwargs.pop("env_path")
  load_dotenv(dotenv_path)
  # Load the environment variables
  repo_code = os.getenv("PCLOUD_CODE")
  
  # Authenticate with the PCloudRepository
  try:
    repo = PCloudRepository(repo_code=repo_code)
    repo.authenticate(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
    )
    set_key(dotenv_path, "PCLOUD_TOKEN", repo.token)
  except ValueError as e:
      if "Authentication failed" in str(e):
          print(f"Error: {e}")
          exit(1)  # Exit the script if authentication fails

  # Set the token in the .env file

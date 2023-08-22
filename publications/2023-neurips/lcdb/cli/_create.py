"""Command line to run experiments."""
import logging
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from lcdb.data import load_task
from lcdb.data.split import train_valid_test_split
from lcdb.utils import import_attr_from_module
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "create"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Generate a list of hyperparameter configurations."
    )

    subparser.add_argument("-w", "--workflow-class", type=str, required=True)
    subparser.add_argument("-n", "--num-configs", type=int, required=True)
    subparser.add_argument(
        "-o", "--output-file", type=str, required=False, default="configs.csv"
    )
    subparser.add_argument(
        "-v", "--verbose", action="store_true", default=False, required=False
    )

    subparser.set_defaults(func=function_to_call)


def main(
    workflow_class,
    num_configs,
    output_file,
    verbose,
):
    """
    :meta private:
    """
    log_dir = os.path.dirname(output_file)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Load the workflow to get its config space
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()

    # Sample the configurations
    # TODO: LHS should be done here
    configs = config_space.sample_configuration(num_configs - 1)

    # Add the default configuration
    config_default = config_space.get_default_configuration()
    configs.insert(0, config_default) # at the beginning

    # Convert the configurations to a dictionnary
    configs = map(lambda c: c.get_dictionary(), configs)

    pd.DataFrame(configs).to_csv(output_file, index=False)

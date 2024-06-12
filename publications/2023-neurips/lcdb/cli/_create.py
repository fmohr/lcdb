"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import os
import pathlib

import pandas as pd
from ..experiments.utils import import_attr_from_module


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
        "-c", "--campaign", type=str, required=False, default=None
    )
    subparser.add_argument(
        "-o", "--output-file", type=str, required=False, default="configs.csv"
    )
    subparser.add_argument(
        "-v", "--verbose", action="store_true", default=False, required=False
    )
    subparser.add_argument("-s", "--seed", type=int, required=False, default=0)

    subparser.set_defaults(func=function_to_call)


def main(
    workflow_class,
    num_configs,
    output_file,
    seed=0,
    verbose=False,
    campaign=None
):
    """
    :meta private:
    """
    from deephyper.problem._hyperparameter import convert_to_skopt_space

    log_dir = os.path.dirname(output_file)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Load the workflow to get its config space
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()

    if verbose:
        print(config_space)

    # Convert the config space to a skopt space
    skopt_space = convert_to_skopt_space(config_space, surrogate_model="RF")

    # Sample the configurations
    # TODO: LHS should be done here
    configs = skopt_space.rvs(n_samples=num_configs - 1, random_state=seed)

    # Add the default configuration
    config_default = config_space.get_default_configuration()
    x = []
    for i, k in enumerate(skopt_space.dimension_names):
        # Check if hyperparameter k is active
        # If it is not active we attribute the "lower bound value" of the space
        # To avoid duplication of the same "entity" in the list of configurations
        if k in config_default.keys():
            val = config_default[k]
        else:
            val = skopt_space.dimensions[i].bounds[0]
        x.append(val)
    configs.insert(0, x)  # at the beginning

    # modify output file
    if campaign is not None:
        if output_file != "configs.csv":
            raise ValueError("You must specify *either* a campaign *or* an output file; in a campaign, the file is always called 'configs.csv' in the respective workflow folder.")
        output_folder = f"{campaign.rstrip('/')}/{workflow_class}"
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file = f"{output_folder}/configs.csv"

    pd.DataFrame(configs, columns=skopt_space.dimension_names).to_csv(
        output_file, index=False
    )

    if verbose:
        print(f"Experiments written to {output_file}")

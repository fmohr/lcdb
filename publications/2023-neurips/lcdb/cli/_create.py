"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import os
import pathlib

import pandas as pd
from ..builder.utils import import_attr_from_module


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
    subparser.add_argument("-ndp", "--num-configs-with-default-preprocessor", type=int, required=False, default=0)
    subparser.add_argument("-ndl", "--num-configs-with-default-learner", type=int, required=False, default=0)
    subparser.add_argument("-ed", "--exclude-default-config", action="store_true", required=False, default=False)
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
    num_configs_with_default_preprocessor,
    num_configs_with_default_learner,
    exclude_default_config,
    output_file,
    seed=0,
    verbose=False,
    campaign=None
):
    """
    :meta private:
    """
    from deephyper.hpo._problem import convert_to_skopt_space

    log_dir = os.path.dirname(output_file)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Load the workflow to get its config space
    WorkflowClass = import_attr_from_module(workflow_class)
    config_space = WorkflowClass.config_space()

    if verbose:
        print(config_space)

    num_fully_random_configs = num_configs - num_configs_with_default_preprocessor - num_configs_with_default_learner
    if num_fully_random_configs < 0:
        raise ValueError(f"num_configs must be at least as high as the sum of num_configs_with_default_preprocessor and num_configs_with_default_learner")

    # Convert the config space to a skopt space
    skopt_space = convert_to_skopt_space(config_space, surrogate_model="RF")

    # Sample the configurations
    # TODO: LHS should be done here
    configs = skopt_space.rvs(n_samples=num_configs if exclude_default_config else num_configs - 1, random_state=seed)

    # create the default configuration
    default_config_raw = config_space.get_default_configuration()
    default_config = []
    for i, k in enumerate(skopt_space.dimension_names):
        # Check if hyperparameter k is active
        # If it is not active we attribute the "lower bound value" of the space
        # To avoid duplication of the same "entity" in the list of configurations
        if k in dict(default_config_raw):
            val = default_config_raw[k]
        else:
            val = skopt_space.dimensions[i].bounds[0]
        default_config.append(val)
    
    # overwrite some of the configs with default learer values
    if num_configs_with_default_learner > 0:

        default_learner_params = {i: v for i, (k, v) in enumerate(zip(skopt_space.dimension_names, default_config)) if not k.startswith("pp@")}
        for c in configs[:num_configs_with_default_learner]:
            for i, v in default_learner_params.items():
                c[i] = v

    # overwrite some of the configs with default learer values
    if num_configs_with_default_preprocessor > 0:

        default_pp_params = {i: v for i, (k, v) in enumerate(zip(skopt_space.dimension_names, default_config)) if k.startswith("pp@")}
        for c in configs[num_configs_with_default_learner:num_configs_with_default_learner + num_configs_with_default_preprocessor]:
            for i, v in default_pp_params.items():
                c[i] = v


    if not exclude_default_config:
        configs.insert(0, default_config)  # at the beginning

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

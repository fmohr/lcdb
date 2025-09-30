"""Command line to create a list of hyperparameter configurations to be evaluated later."""
import os
import pathlib

import pandas as pd
from lcdb.workflow._util import get_config_space_of_workflow


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
    config_space = get_config_space_of_workflow(workflow_class)

    if verbose:
        print(config_space)

    num_fully_random_configs = num_configs - num_configs_with_default_preprocessor - num_configs_with_default_learner
    if num_fully_random_configs < 0:
        raise ValueError(f"num_configs must be at least as high as the sum of num_configs_with_default_preprocessor and num_configs_with_default_learner")

    # Convert the config space to a skopt space
    #skopt_space = convert_to_skopt_space(config_space, surrogate_model="RF")

    # Sample the configurations
    # TODO: LHS should be done here
    config_space.seed(seed)
    configs = [dict(c) for c in config_space.sample_configuration(size=num_configs if exclude_default_config else num_configs - 1)]
    for config in configs:

        # make sure that the value for feature gen is always set
        if not "pp@featuregen" in config:
            config["pp@featuregen"] = "none"
    if verbose:
        for config in configs:
            print(config)

    # get default config as basis to modify other configs
    config_default = config_space.get_default_configuration()

    # Add the default configuration if it is not excluded
    if not exclude_default_config:
        configs.insert(0, config_default)  # at the beginning

    # overwrite some of the configs with default learner values
    cols_no_pp = [c for c in config_space.keys() if "pp@" not in c]
    if num_configs_with_default_learner > 0:
        default_learner_params = {hp: config_space[hp].default_value for hp in cols_no_pp}
        for c in configs[:num_configs_with_default_learner]:
            if isinstance(c, dict):
                c.update(default_learner_params)
            else:
                for k, v in default_learner_params.items():
                    c[k] = v


    # overwrite some of the configs with default pre-processor values
    cols_pp = [c for c in config_space.keys() if "pp@" in c]
    if num_configs_with_default_preprocessor > 0:
        default_preprocessor_params = {hp: config_space[hp].default_value for hp in cols_pp}
        for c in configs[num_configs_with_default_learner:num_configs_with_default_learner + num_configs_with_default_preprocessor]:
            c.update(default_preprocessor_params)

    # modify output file
    if campaign is not None:
        if output_file != "configs.csv":
            raise ValueError("You must specify *either* a campaign *or* an output file; in a campaign, the file is always called 'configs.csv' in the respective workflow folder.")
        output_folder = f"{campaign.rstrip('/')}/{workflow_class}"
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file = f"{output_folder}/configs.csv"

    
    pd.DataFrame(configs,columns=cols_pp + cols_no_pp).astype(
        {c: "Int64" for c in ["pp@selectp_percentile", "pp@poly_degree", "pp@feature_map_size"]}
    ).to_csv(
        output_file, index=False
    )
    if verbose:
        print(f"Experiments written to {output_file}")

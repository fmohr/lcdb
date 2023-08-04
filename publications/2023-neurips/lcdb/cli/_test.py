"""Command line to test the default configuration of a workflow."""

import json
import logging

from ..workflow._util import get_all_experiments, run


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "test"
    function_to_call = main

    subparser = subparsers.add_parser(subparser_name, help="Test a worfklow.")

    subparser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    subparser.add_argument(
        "--num-configs",
        type=int,
        required=False,
        default=1,
        help="The number of hyperparameter configurations that are being sampled (including the default configuration).",
    )
    subparser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="The random seed used in sampling configurations.",
    )
    subparser.add_argument(
        "--max-num_anchors-per-row",
        type=int,
        required=False,
        default=1,
        help="The number of anchors per row in the database.",
    )
    subparser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Whether to print the results of the experiments.",
    )
    subparser.set_defaults(func=function_to_call)


def main(
    config: str,
    num_configs: int,
    seed: int,
    max_num_anchors_per_row: int,
    verbose: bool,
    *args,
    **kwargs
):
    """
    :meta private:
    """

    if verbose:
        logging.basicConfig(
            # filename=path_log_file, # optional if we want to store the logs to disk
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

    # create experiment rows
    workflow_class, experiments = get_all_experiments(
        config_file=config,
        num_configs=num_configs,
        seed=seed,
        max_num_anchors_per_row=max_num_anchors_per_row,
        LHS=False,
    )

    # filter experiments
    if hasattr(workflow_class, "is_experiment_valid"):
        experiments = [e for e in experiments if workflow_class.is_experiment_valid(e)]

    # replace hyperparameters by strings
    for e in experiments:
        e["hyperparameters"] = json.dumps(e["hyperparameters"])
        e["train_sizes"] = json.dumps(e["train_sizes"])

        logging.info("Running experiment %s", e)

        results = run(
            openmlid=int(e["openmlid"]),
            workflow_class=workflow_class,
            anchors=json.loads(e["train_sizes"]),
            monotonic=bool(e["monotonic"]),
            inner_seed=int(e["seed_inner"]),
            outer_seed=int(e["seed_outer"]),
            hyperparameters=json.loads(e["hyperparameters"]),
            logger=logging,
        )
        logging.info("Results: %s", results)

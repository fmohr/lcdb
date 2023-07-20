"""Command line to create/generate new experiments."""

import json

from ..workflow._util import get_all_experiments, get_experimenter


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "create"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Create new experiments from a configuration file."
    )

    subparser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    subparser.add_argument(
        "--num_configs",
        type=int,
        required=False,
        default=10,
        help="The number of hyperparameter configurations that are being sampled.",
    )
    subparser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="The random seed used in sampling configurations.",
    )
    subparser.add_argument(
        "--max_num_anchors_per_row",
        type=int,
        required=False,
        default=3,
        help="The number of anchors per row in the database.",
    )
    subparser.set_defaults(func=function_to_call)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def main(
    config: str,
    num_configs: int,
    seed: int,
    max_num_anchors_per_row: int,
    *args,
    **kwargs
):
    """
    :meta private:
    """

    # create experiment rows
    workflow_class, experiments = get_all_experiments(
        config_file=config,
        num_configs=num_configs,
        seed=seed,
        max_num_anchors_per_row=max_num_anchors_per_row,
    )

    # filter experiments
    if hasattr(workflow_class, "is_experiment_valid"):
        experiments = [e for e in experiments if workflow_class.is_experiment_valid(e)]

    # replace hyperparameters by strings
    for e in experiments:
        e["hyperparameters"] = json.dumps(e["hyperparameters"])
        e["train_sizes"] = json.dumps(e["train_sizes"])

    # create all rows for the experiments
    print(list(experiments[0].keys()))

    print('total experiments: %d ' % len(experiments))

    batch_size = 10000
    batches = batch(experiments, batch_size)
    num_batches = len(experiments) / batch_size
    for (cur_batch_num, B) in enumerate(batches):
        print('inserting batch %d of %d...' % (cur_batch_num, num_batches))
        get_experimenter(config_file=config).fill_table_with_rows(rows=B)

"""Command line to run experiments."""

import json
import logging

from py_experimenter.result_processor import ResultProcessor

from ..workflow._util import run, import_attr_from_module, get_experimenter

logger = logging.getLogger("lcdb.exp")
logger.setLevel(logging.DEBUG)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "run"
    function_to_call = main

    subparser = subparsers.add_parser(subparser_name, help="Run experiments.")

    subparser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    subparser.add_argument(
        "--executor-name",
        type=str,
        required=True,
        help="Name of the executor. Used for debugging.",
    )

    subparser.set_defaults(func=function_to_call)


def run_experiment(
    keyfields: dict, result_processor: ResultProcessor, custom_config: dict
):
    # activate logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(ch)

    results = run(
        openmlid=int(keyfields["openmlid"]),
        workflow_class=import_attr_from_module(keyfields["workflow"]),
        anchors=json.loads(keyfields["train_sizes"]),
        monotonic=bool(keyfields["monotonic"]),
        inner_seed=int(keyfields["seed_inner"]),
        outer_seed=int(keyfields["seed_outer"]),
        hyperparameters=json.loads(keyfields["hyperparameters"]),
        logger=logger,
    )

    # unpack results
    resultfields = {"result": {}}
    for anchor, results_for_anchor in results.items():
        if type(results_for_anchor) is not dict:
            resultfields["result"][anchor] = f"Exception: {results_for_anchor}"
        else:
            resultfields["result"][anchor] = json.dumps(results_for_anchor)

    # Write intermediate results to database
    result_processor.process_results(resultfields)


def main(config: str, executor_name: str, *args, **kwargs):
    """
    :meta private:
    """

    experimenter = get_experimenter(config_file=config, executor_name=executor_name)

    experimenter.execute(run_experiment, -1)

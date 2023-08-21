"""Command line to run experiments."""

import json
import logging
import os
from time import time
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
        default='defaultmachine',
        required=False,
        help="Name of the executor. Used for debugging.",
    )

    subparser.add_argument(
        "--num",
        type=int,
        default=-1,
        required=False,
        help="Number of configurations to run"
    )

    subparser.set_defaults(func=function_to_call)


def run_experiment(
    keyfields: dict, result_processor: ResultProcessor, custom_config: dict
):

    print('**** starting experiment on process id %d ****' % os.getpid())
    print(time())
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

    print(keyfields)

    results, postprocess = run(
        openmlid=int(keyfields["openmlid"]),
        workflow_class=import_attr_from_module(keyfields["workflow"]),
        anchors=json.loads(keyfields["train_sizes"]),
        monotonic=bool(keyfields["monotonic"]),
        inner_seed=int(keyfields["seed_inner"]),
        outer_seed=int(keyfields["seed_outer"]),
        hyperparameters=json.loads(keyfields["hyperparameters"]),
        maxruntime=int(keyfields["maxruntime"]),
        valid_prop=float(keyfields["valid_prop"]),
        test_prop=float(keyfields["test_prop"]),
        measure_memory=bool(keyfields["measure_memory"]),
        logger=logger,
    )

    resultfields = {}
    resultfields["result"] = json.dumps(results)
    resultfields["postprocess"] = postprocess
    # Write intermediate results to database
    result_processor.process_results(resultfields)


def main(config: str, executor_name: str, num:int, *args, **kwargs):
    """
    :meta private:
    """

    experimenter = get_experimenter(config_file=config, executor_name=executor_name)

    ts_fit_start = time()
    experimenter.execute(run_experiment, num)
    ts_fit_end = time()
    run_time = ts_fit_end - ts_fit_start
    print('Took %d seconds' % run_time)


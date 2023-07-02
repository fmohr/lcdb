"""Command line to run experiments."""

import json
import logging
import sys

from py_experimenter.result_processor import ResultProcessor

from ..workflow._util import get_experimenter, run

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
        "--workflow", type=str, required=True, help="Name of workflow class."
    )
    subparser.add_argument(
        "--executor_name",
        type=str,
        required=True,
        help="Name of the executor. Used for debugging.",
    )

    subparser.set_defaults(func=function_to_call)


def get_workflow_class_from_name(name):
    return getattr(sys.modules["lcdb.workflow"], name)


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
    # logger.addHandler(ch)

    results = run(
        openmlid=int(keyfields["openmlid"]),
        workflow_class=get_workflow_class_from_name(custom_config["workflow_class"]),
        anchors=json.loads(keyfields["train_sizes"]),
        monotonic=bool(keyfields["monotonic"]),
        inner_seed=int(keyfields["seed_inner"]),
        outer_seed=int(keyfields["seed_outer"]),
        hyperparameters=json.loads(keyfields["hyperparameters"]),
        logger=logger,
    )

    # unpack results
    resultfields = {
        "result": {}
    }
    for anchor, results_for_anchor in results.items():
        if type(results_for_anchor) is not tuple:
            resultfields["result"][anchor] = f"Exception: {results_for_anchor}"
        else:
            (
                labels,
                cm_train,
                cm_valid,
                cm_test,
                cm_valid_sub,
                cm_test_sub,
                fit_time,
                predict_time_train,
                predict_time_valid,
                predict_time_test,
                additional_log_data,
            ) = results_for_anchor
            resultfields["result"][anchor] = json.dumps(
                [
                    labels,
                    cm_train.tolist(),
                    cm_valid.tolist(),
                    cm_test.tolist(),
                    cm_valid_sub.tolist(),
                    cm_test_sub.tolist(),
                    fit_time,
                    predict_time_train,
                    predict_time_valid,
                    predict_time_test,
                    additional_log_data,
                ]
            )

    # Write intermediate results to database
    result_processor.process_results(resultfields)


def main(workflow: str, executor_name: str, *args, **kwargs):
    """
    :meta private:
    """

    # get workflow class
    workflow_class = get_workflow_class_from_name(workflow)

    experimenter = get_experimenter(
        workflow_class, executor_name=executor_name, config_folder="config"
    )

    experimenter.execute(run_experiment, -1)

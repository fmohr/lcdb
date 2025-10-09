import logging

from parameterized import parameterized
import unittest

from lcdb.analysis.views._debugger import Debugger
import pandas as pd
from pathlib import Path
import shutil
from lcdb.cli._cli import create_parser
import itertools as it
import json

from lcdb.builder.utils import StatusFileManager, deephyper_results_to_jsonl

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

DATASETS = [
    61,
    188,
    1596
]


WORKFLOWS = [
    "lcdb.workflow.sklearn.KNNWorkflow",
    "lcdb.workflow.sklearn.LibLinearWorkflow",
    "lcdb.workflow.sklearn.LibSVMWorkflow",
    "lcdb.workflow.sklearn.TreesEnsembleWorkflow",
    "lcdb.workflow.xgboost.XGBoostWorkflow",
    "lcdb.workflow.keras.DenseNNWorkflow"
]

VAL_SEEDS = [0]
TEST_SEEDS = [0]
WORKFLOW_SEEDS = [0]

MAX_SAMPLE_ANCHOR = 512


class TestRunFunctionalities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(__file__).parent
        cls.num_evals_in_first = 4
        cls.test_status_dir = Path(f"{cls.folder}/test_status_dir")
        if cls.test_status_dir.exists():
            shutil.rmtree(cls.test_status_dir)

    @parameterized.expand(list(it.product(WORKFLOWS, DATASETS)))
    def test_base_run(self, workflow, openmlid):

        logger.info(f"Test lcdb run for -w {workflow} on dataset {openmlid}")

        # prepare folder structure
        test_status_dir_for_case = Path(f"{__class__.test_status_dir}/{workflow}_{openmlid}")

        # create CLI parser
        parser = create_parser()

        # define params
        params = {}
        if "TreesEnsemble" in workflow:
            params["n_estimators"] = 8
        if "xgboost" in workflow or "keras" in workflow in workflow:
            params["num_epochs"] = 8
            if "keras" in workflow: # overwrite the randomly sampled network design by a small architecture (architecture is not subject to test here)
                params["num_layers"] = 3
                params["num_units_first"] = 16
                params["num_units_last"] = 8

        # parse command
        args = parser.parse_args(
            f"run"
            f" -i {openmlid}"
            f" -w {workflow}"
            f" --log-dir={test_status_dir_for_case}"
            f" --status-dir={test_status_dir_for_case}"
            f" --initial-configs={__class__.folder}/run_configs/{workflow}.csv"
            f" --max-evals={__class__.num_evals_in_first}"
            f" --anchor-schedule=first"
            f" --epoch-schedule=first"
            f" --parameters={json.dumps(params).replace(' ', '')}"
            " -e serial"
            " --no-exception-on-unsuitable-preprocessor"
        .split())

        # execute command
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)

        # rewrite results
        deephyper_results_to_jsonl(f"{test_status_dir_for_case}/results.csv", f"{test_status_dir_for_case}/results.jsonl")

        # check that there are no errors
        debugger = Debugger()
        debugger.load_data(jsonl=f"{test_status_dir_for_case}/results.jsonl")
        self.assertEqual(__class__.num_evals_in_first, debugger.num_rows)

        # remove some of the error files if they are expected
        if "LibSVM" in workflow and debugger.num_rows > 0:
            debugger.reduce(lambda r: any(["The dual coefficients or intercepts are not finite." in s["message"] for s in r["traceback_summary"]]) if r["traceback_summary"] is not None else True)
        if "keras" in workflow and debugger.num_rows > 0:
            debugger.reduce(lambda r: any(["There are NAN values in the NN prediction." in s["message"] for s in r["traceback_summary"]]) if r["traceback_summary"] is not None else True)

        # check that no unexpected errors are left
        error_messages = debugger.get_error_messages()
        self.assertEqual(0, len(error_messages), msg=f"There should be no errors, but we observed these error messages: {error_messages}")

    def test_resume_knn_run(self):
        logger.info("Test resume KNN")
        
        workflow = "lcdb.workflow.sklearn.KNNWorkflow"
        openmlid = 61

        # prepare folder structure
        test_status_dir_for_case = Path(f"{__class__.test_status_dir}/{workflow}_{openmlid}")

        for round in range(1, 3):

            logger.info(f"Additional Round #{round}")

            # remove status files for running
            status_file_manager = StatusFileManager(test_status_dir_for_case)
            for status in [StatusFileManager.EXPERIMENT_STATUS_COMPLETED, StatusFileManager.EXPERIMENT_STATUS_RUNNING]:
                status_file_manager.remove_status_file(
                    workflow=workflow,
                    openmlid=openmlid,
                    campaign="default_campaign",
                    workflowseed=42,
                    valseed=42,
                    testseed=42,
                    status=status
                )

            # create CLI parser
            parser = create_parser()

            # parse command
            num_evals = __class__.num_evals_in_first + 2 * round
            args = parser.parse_args(
                f"run"
                f" -i {openmlid}"
                f" -w {workflow}"
                f" --log-dir={test_status_dir_for_case}"
                f" --status-dir={test_status_dir_for_case}"
                f" --initial-configs={__class__.folder}/run_configs/{workflow}.csv"
                f" --max-evals={num_evals}"
                f" --anchor-schedule=first"
                f" --epoch-schedule=first"
                " --log-level=debug"
                " -e serial"
                " --no-exception-on-unsuitable-preprocessor"
            .split())

            # execute command
            logger.debug("Starting Search")
            func = args.func
            kwargs = vars(args)
            kwargs.pop("func")
            func(**kwargs)

            # check whether results have arrived without error
            df_results = pd.read_csv(f"{test_status_dir_for_case}/results.csv")
            self.assertEqual(num_evals, len(df_results))
import logging

from parameterized import parameterized
import unittest

from lcdb.db.processors._traceback_extractor import TracebackExtractor
from lcdb.db._results import ResultSet
import pandas as pd
from pathlib import Path
import shutil
from lcdb.cli._cli import create_parser
import itertools as it
import json
import time

from lcdb.builder.utils import StatusFileManager, deephyper_results_to_jsonl

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

lcdb_logger = logging.getLogger("LCDB")
lcdb_logger.handlers.clear()
lcdb_logger.addHandler(ch)
lcdb_logger.setLevel(logging.WARNING)

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
        cls.num_evals_in_first = 2
        cls.test_status_dir = Path(f"{cls.folder}/test_status_dir")
        if cls.test_status_dir.exists():
            shutil.rmtree(cls.test_status_dir)

    @parameterized.expand(list(it.product(WORKFLOWS, DATASETS)))
    def test_1_base_run(self, workflow, openmlid):

        logger.info(f"Test lcdb run for -w {workflow} on dataset {openmlid}")

        # prepare folder structure
        test_status_dir_for_case = Path(f"{__class__.test_status_dir}/{workflow}_{openmlid}")

        # create CLI parser
        parser = create_parser()

        # define params
        params = {}
        if "TreesEnsemble" in workflow:
            params["n_estimators"] = 3
        if "xgboost" in workflow or "keras" in workflow in workflow:
            params["num_epochs"] = 3
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

        # check that there are no errors
        rs = ResultSet()
        rs.load(f"{test_status_dir_for_case}/results.jsonl")
        rs._unpack_build_issues()
        rs.apply(TracebackExtractor())
        self.assertEqual(__class__.num_evals_in_first, rs.num_rows)

        # remove some of the error files if they are expected
        if "LibSVM" in workflow and rs.num_rows > 0:
            rs.reduce(lambda r: any(["The dual coefficients or intercepts are not finite." in s["message"] for s in r["traceback_summary"]]) if r["traceback_summary"] is not None else True)
        if "keras" in workflow and rs.num_rows > 0:
            rs.reduce(lambda r: any(["There are NAN values in the NN prediction." in s["message"] for s in r["traceback_summary"]]) if r["traceback_summary"] is not None else True)

        # check that no unexpected errors are left
        rs.drop_rows_without_build_issues()
        self.assertEqual(0, rs.num_rows, msg=f"There should be no errors, but we observed {rs.num_rows} rows with build errors left.")

    def test_4_resume_knn_run(self):
        logger.info("Test resume KNN")
        
        workflow = "lcdb.workflow.sklearn.KNNWorkflow"
        openmlid = 61

        # prepare folder structure
        test_status_dir_for_case = Path(f"{__class__.test_status_dir}/{workflow}_{openmlid}")

        multipliers = [1, 2, 3, 3] # intentionally run the last one twice to see whether we get the result file again
        for round, multiplier in zip(range(1, len(multipliers) + 1), multipliers):

            logger.info(f"Round #{round}")

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
            num_evals = __class__.num_evals_in_first + 2 * multiplier
            args_raw = (
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
            )
            args = parser.parse_args(args_raw.split())

            # execute command
            t_start = time.time()
            logger.debug(f"Executing lcdb {args_raw}")
            func = args.func
            kwargs = vars(args)
            kwargs.pop("func")
            func(**kwargs)
            runtime = time.time() - t_start

            if round == 4:
                assert runtime < 0.1, "In the last round, we should only pick up results and re-write them. This should be blazingly fast (less than 100ms)."
            else:
                time.sleep(0.5) # we sleep 500ms to avoid that result files generated by deephyper have identical names

            # check whether results have arrived without error
            result_file = Path(f"{test_status_dir_for_case}/results.jsonl")
            rs = ResultSet.read_jsonl(result_file)
            self.assertEqual(num_evals, rs.num_rows)

            # now check whether we get the same result file again without running anything
            result_file.unlink()
            assert not result_file.exists()

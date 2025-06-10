import pytest
import unittest

import numpy as np
from lcdb.analysis import LearningCurveExtractor, merge_curves
from lcdb import LCDB, Debugger
from parameterized import parameterized
import logging

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("LCDB")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


@pytest.mark.db
class TestExtractors(unittest.TestCase):

    @parameterized.expand(
        [
            (1111, "lcdb.workflow.sklearn.KNNWorkflow", 0, 0, 42),
            (3, "lcdb.workflow.sklearn.LibLinearWorkflow", 0, 0, 42),
            (3, "lcdb.workflow.sklearn.LibSVMWorkflow", 0, 0, 42),
            (1111, "lcdb.workflow.sklearn.TreesEnsembleWorkflow", 0, 0, 42),
        ]
    )
    def test_learning_curve_extraction(
        self, openmlid, workflow, val_seed, test_seed, workflow_seed
    ):

        logger.info(f"Starting test for extraction on {openmlid=}, {workflow=}, {val_seed=}, {test_seed=}, {workflow_seed=}.")
        metrics = ["error_rate"]  # , "balanced_error_rate"]

        lcdb = LCDB()
        df = lcdb.query(
            openmlids=[openmlid],
            workflows=workflow,
            return_generator=False,
            test_seeds=[test_seed],
            validation_seeds=[val_seed],
            workflow_seeds=[workflow_seed],
            processors={
                "learning_curve": LearningCurveExtractor(
                    metrics=metrics, folds=["train", "val", "test", "oob"]
                )
            },
            show_progress=True,
        )

        oob_fold_expected = "TreesEnsembleWorkflow" in workflow
        num_oob_not_nan = 0
        print(df)
        self.assertIsNotNone(df, f"No results found for {openmlid=}, {workflow=}, {val_seed=}, {test_seed=}, {workflow_seed=}")
        
        df = df[df["has_result"]] # filter out configs for which no results are available

        # test that all curves are proper
        for lc in df["learning_curve"]:
            self.assertEqual(len(metrics), lc.values.shape[0])
            self.assertEqual(4, lc.values.shape[1])
            if lc.is_iteration_wise_curve:
                if not np.isnan(lc.values[0, 3, 0, 0, 0, 0, 0]):
                    num_oob_not_nan += 1
            else:
                if not np.isnan(lc.values[0, 3, 0, 0, 0, 0]):
                    num_oob_not_nan += 1

        self.assertTrue(not oob_fold_expected or num_oob_not_nan > 0)

    @parameterized.expand(
        [
            # (6, "lcdb.workflow.sklearn.KNNWorkflow"),
            (3, "lcdb.workflow.sklearn.LibLinearWorkflow"),
            (3, "lcdb.workflow.sklearn.LibSVMWorkflow"),
            # (6, "lcdb.workflow.sklearn.TreesEnsembleWorkflow")
        ]
    )
    def test_learning_curve_grouping_after_extraction(self, openmlid, workflow):

        campaigns = ["snellius"]
        validation_seeds = [0, 1]
        test_seeds = [0]
        logger.info(
            f"Testing extraction of learning curves on dataset {openmlid} for workflow {workflow}"
            f"Considered campaigns: {campaigns}, validation seeds: {validation_seeds}, test_seeds: {test_seeds}"
        )
        lcdb = LCDB()
        df = lcdb.query(
            campaigns=campaigns,
            openmlids=[openmlid],
            workflows=workflow,
            return_generator=False,
            test_seeds=test_seeds,
            validation_seeds=validation_seeds,
            processors={
                "learning_curve": LearningCurveExtractor(
                    metrics=["error_rate"], folds=["train", "val", "test", "oob"]
                )
            },
            show_progress=True,
        )

        if df is not None:

            config_cols = [c for c in df.columns if c.startswith("p:")]
            len_before = len(df)
            len_after = len(
                df.groupby(config_cols).agg({"learning_curve": merge_curves})
            )
            self.assertEqual(len_before, len_after * len(validation_seeds))


@pytest.mark.db
class TestDebugger(unittest.TestCase):

    def test_cli_command_extraction(self):

        openmlid = 1111
        workflow = "lcdb.workflow.sklearn.LibLinearWorkflow"

        debugger = Debugger()
        debugger.load_data(
            workflows=[workflow],
            openmlids=[openmlid],
            show_progress=True
        )
        self.assertTrue(len(debugger.df) > 0)
        for cmd in debugger.get_cli_test_command():
            self.assertTrue(cmd.startswith(f"lcdb test -i {openmlid} -w {workflow} --parameters='"))

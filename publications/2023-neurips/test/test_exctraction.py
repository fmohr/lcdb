from lcdb.db import LCDB

from parameterized import parameterized
import unittest

import numpy as np
from lcdb.analysis.util import LearningCurveExtractor


class TestExtractors(unittest.TestCase):

    @parameterized.expand([
        (6, "lcdb.workflow.sklearn.KNNWorkflow", 0, 0, 42),
        (3, "lcdb.workflow.sklearn.LibLinearWorkflow", 0, 0, 42),
        (3, "lcdb.workflow.sklearn.LibSVMWorkflow", 0, 0, 42),
        (6, "lcdb.workflow.sklearn.TreesEnsembleWorkflow", 0, 1, 42)
    ])
    def test_learning_curve_extraction(self, openmlid, workflow, val_seed, test_seed, workflow_seed):

        lcdb = LCDB()
        df = lcdb.query(
            openmlids=[openmlid],
            workflows=workflow,
            return_generator=False,
            test_seeds=[test_seed],
            validation_seeds=[val_seed],
            workflow_seeds=[workflow_seed],
            processors={
                "learning_curve":LearningCurveExtractor(
                    metrics=["error_rate", "balanced_error_rate"],
                    folds=["train", "val", "test", "oob"]
                )
            },
            show_progress=True
        )

        oob_fold_expected = "TreesEnsembleWorkflow" in workflow
        num_oob_not_nan = 0
        if df is not None:
            for lc in df["learning_curve"]:
                self.assertEqual(2, lc.values.shape[0])
                self.assertEqual(4, lc.values.shape[1])
                if lc.is_iteration_curve:
                    if not np.isnan(lc.values[0, 3, 0, 0]):
                        num_oob_not_nan += 1
                else:
                    if not np.isnan(lc.values[0, 3, 0]):
                        num_oob_not_nan += 1
        self.assertTrue(not oob_fold_expected or num_oob_not_nan > 0)
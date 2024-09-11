import logging

from lcdb.db import LCDB

from parameterized import parameterized
import unittest

import numpy as np
from lcdb.analysis.util import LearningCurveExtractor, merge_curves
from lcdb.builder import run_learning_workflow
from lcdb.analysis.json import QueryPreprocessorResults


class TestExtractors(unittest.TestCase):

    @parameterized.expand([
        (6, "lcdb.workflow.sklearn.KNNWorkflow", 0, 0, 42),
        (3, "lcdb.workflow.sklearn.LibLinearWorkflow", 0, 0, 42),
        (3, "lcdb.workflow.sklearn.LibSVMWorkflow", 0, 0, 42),
        (6, "lcdb.workflow.sklearn.TreesEnsembleWorkflow", 0, 1, 42)
    ])
    def test_learning_curve_extraction(self, openmlid, workflow, val_seed, test_seed, workflow_seed):

        metrics = ["error_rate"]#, "balanced_error_rate"]

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
                    metrics=metrics,
                    folds=["train", "val", "test", "oob"]
                )
            },
            show_progress=True
        )

        oob_fold_expected = "TreesEnsembleWorkflow" in workflow
        num_oob_not_nan = 0
        if df is not None:

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

    @parameterized.expand([
        #(6, "lcdb.workflow.sklearn.KNNWorkflow"),
        (3, "lcdb.workflow.sklearn.LibLinearWorkflow"),
        (3, "lcdb.workflow.sklearn.LibSVMWorkflow"),
        #(6, "lcdb.workflow.sklearn.TreesEnsembleWorkflow")
    ])
    def test_learning_curve_grouping_after_extraction(self, openmlid, workflow):

        lcdb = LCDB()
        validation_seeds = [0, 1, 2, 3, 4]
        df = lcdb.query(
            openmlids=[openmlid],
            workflows=workflow,
            return_generator=False,
            test_seeds=[0],
            validation_seeds=validation_seeds,
            processors={
                "learning_curve": LearningCurveExtractor(
                    metrics=["error_rate"],
                    folds=["train", "val", "test", "oob"]
                )
            },
            show_progress=True
        )

        if df is not None:

            config_cols = [c for c in df.columns if c.startswith("p:")]
            len_before = len(df)
            len_after = len(df.groupby(config_cols).agg({"learning_curve": merge_curves}))
            self.assertEqual(len_before, len_after * len(validation_seeds))

    @parameterized.expand([
        (11, "lcdb.workflow.sklearn.KNNWorkflow"),
        (3, "lcdb.workflow.sklearn.LibLinearWorkflow"),
        (3, "lcdb.workflow.sklearn.LibSVMWorkflow")
    ])
    def test_preprocessing_usage(self, openmlid, workflow):

        from deephyper.evaluator import RunningJob
        from lcdb.builder.utils import import_attr_from_module, terminate_on_memory_exceeded

        WorkflowClass = import_attr_from_module(workflow)
        config_space = WorkflowClass.config_space()
        config = dict(config_space.get_default_configuration())

        config.update({
            "p:metric": "nan_euclidean",
            "p:n_neighbors": 28,
            "p:pp@cat_encoder": "onehot",
            "p:pp@decomposition": "kernel_pca",
            "p:pp@featuregen": "poly",
            "p:pp@featureselector": "selectp",
            "p:pp@scaler": "minmax",
            "p:weights": "distance",
            "p:p": 1,
            "p:pp@kernel_pca_kernel": "linear",
            "p:pp@kernel_pca_n_components": 0.25,
            "p:pp@poly_degree": 2,
            "p:pp@selectp_percentile": 25,
            "p:pp@std_with_std": True
        })

        # define stream handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger = logging.getLogger("LCDB")
        logger.addHandler(ch)

        output = run_learning_workflow(
            RunningJob(id=0, parameters=config),
            openml_id=openmlid,
            workflow_class=workflow,
            task_type="classification",
            monotonic=False,
            valid_seed=0,
            test_seed=0,
            workflow_seed=0,
            valid_prop=0.1,
            test_prop=0.1,
            timeout_on_fit=60,
            anchor_schedule="power",
            epoch_schedule=None,
            logger=logger
        )

        for executed_preprocessors in QueryPreprocessorResults()(output["metadata"]["json"]):
            self.assertEqual(5, len(executed_preprocessors))

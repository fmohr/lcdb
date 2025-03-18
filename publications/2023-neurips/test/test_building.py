import logging

from parameterized import parameterized
import unittest

from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from lcdb.builder import run_learning_workflow
from lcdb.builder.utils import import_attr_from_module
import itertools as it

from lcdb.analysis.json import QueryPreprocessorResults, QueryDatasetMetadata, QueryAnchorValues
import numpy as np

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("LCDB")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

DATASETS = [
    61,
    3,
    188,
    6,
]

WORKFLOWS = [
    "lcdb.workflow.sklearn.GaussianNBWorkflow",
    "lcdb.workflow.sklearn.LDAWorkflow",
    "lcdb.workflow.sklearn.QDAWorkflow",
    "lcdb.workflow.sklearn.KNNWorkflow",
    "lcdb.workflow.sklearn.LRWorkflow",
    "lcdb.workflow.sklearn.RidgeWorkflow",
    "lcdb.workflow.sklearn.PAWorkflow",
    "lcdb.workflow.sklearn.PerceptronWorkflow",
    "lcdb.workflow.sklearn.LibLinearWorkflow",
    "lcdb.workflow.sklearn.LibSVMWorkflow",
    "lcdb.workflow.sklearn.MajorityWorkflow",
    "lcdb.workflow.sklearn.RandomWorkflow",
    "lcdb.workflow.sklearn.DTWorkflow",
    "lcdb.workflow.sklearn.TreesEnsembleWorkflow",
    "lcdb.workflow.xgboost.XGBoostWorkflow"
]

VAL_SEEDS = [0]
TEST_SEEDS = [0]
WORKFLOW_SEEDS = [0]


class TestBuildFunctionalities(unittest.TestCase):

    @parameterized.expand(list(it.product([61], WORKFLOWS, VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS)))
    def test_workflow_base_functionality_and_integrity(self, openmlid, workflow, val_seed, test_seed, workflow_seed):

        workflow_class = import_attr_from_module(workflow)

        if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
            params = {
                "pp@cat_encoder": "onehot"
            }
        else:
            params = None

        logger.info(f"Starting test of workflow {workflow} on dataset {openmlid}")
        try:
            out = run_learning_workflow(
                openml_id=openmlid,
                workflow_class=workflow,
                workflow_parameters=params,
                valid_seed=val_seed,
                test_seed=test_seed,
                workflow_seed=workflow_seed,
                raise_errors=True,
                anchor_schedule="power-2-2-2",
                epoch_schedule="power-2-2-2"
            )

            final_node = out["metadata"]["json"]["children"][-1]
            self.assertEqual("build_curves", final_node["tag"])
            first_anchor_in_final_node = final_node["children"][0]
            self.assertEqual("anchor", first_anchor_in_final_node["tag"])
            self.assertEqual(64, first_anchor_in_final_node["metadata"]["value"])
            metrics_in_first_anchor_in_final_node = first_anchor_in_final_node["children"][-1]
            self.assertEqual("metrics", metrics_in_first_anchor_in_final_node["tag"])
            validation_confusion_matrix_in_first_anchor_in_final_node = metrics_in_first_anchor_in_final_node["children"][1]["children"][0]
            self.assertEqual("confusion_matrix", validation_confusion_matrix_in_first_anchor_in_final_node["tag"])

            def test_timestamp_consistency(d, earliest_ts_start=0):
                ts_start = d["timestamp_start"]
                ts_end = d["timestamp_stop"]
                self.assertTrue(ts_start >= earliest_ts_start)
                self.assertTrue(ts_start <= ts_end)

                # test integrity of children
                if "children" in d:
                    t_cur = ts_start
                    for child in d["children"]:
                        t_cur = test_timestamp_consistency(child, earliest_ts_start=t_cur)
                    self.assertTrue(t_cur <= ts_end)
                return ts_end

            test_timestamp_consistency(out["metadata"]["json"])

        except Exception as e:
            msg = str(e)
            if "covariance is ill defined" in msg:
                pass
            else:
                raise e

    @parameterized.expand(list(it.product(DATASETS, WORKFLOWS[:2], VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS)))
    def test_ability_to_work_all_types_of_datasets(self, openmlid, workflow, val_seed, test_seed, workflow_seed):

        workflow_class = import_attr_from_module(workflow)

        if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
            params = {
                "pp@cat_encoder": "onehot"
            }
        else:
            params = None

        logger.info(f"Starting test of workflow {workflow} on dataset {openmlid}")
        try:
            out = run_learning_workflow(
                openml_id=openmlid,
                workflow_class=workflow,
                workflow_parameters=params,
                valid_seed=val_seed,
                test_seed=test_seed,
                workflow_seed=workflow_seed,
                raise_errors=True,
                anchor_schedule="power-2-2-2",
                epoch_schedule="power-2-2-2"
            )

            final_node = out["metadata"]["json"]["children"][-1]
            self.assertEqual("build_curves", final_node["tag"])
            first_anchor_in_final_node = final_node["children"][0]
            self.assertEqual("anchor", first_anchor_in_final_node["tag"])
            self.assertEqual(64, first_anchor_in_final_node["metadata"]["value"])
            metrics_in_first_anchor_in_final_node = first_anchor_in_final_node["children"][-1]
            self.assertEqual("metrics", metrics_in_first_anchor_in_final_node["tag"])
            validation_confusion_matrix_in_first_anchor_in_final_node = metrics_in_first_anchor_in_final_node["children"][1]["children"][0]
            self.assertEqual("confusion_matrix", validation_confusion_matrix_in_first_anchor_in_final_node["tag"])

            def test_timestamp_consistency(d, earliest_ts_start=0):
                ts_start = d["timestamp_start"]
                ts_end = d["timestamp_stop"]
                self.assertTrue(ts_start >= earliest_ts_start)
                self.assertTrue(ts_start <= ts_end)

                # test integrity of children
                if "children" in d:
                    t_cur = ts_start
                    for child in d["children"]:
                        t_cur = test_timestamp_consistency(child, earliest_ts_start=t_cur)
                    self.assertTrue(t_cur <= ts_end)
                return ts_end

            test_timestamp_consistency(out["metadata"]["json"])

        except Exception as e:
            msg = str(e)
            if "covariance is ill defined" in msg:
                pass
            else:
                raise e

    @parameterized.expand([
        (3, "lcdb.workflow.sklearn.KNNWorkflow", 25, 40, 9),
        (188, "lcdb.workflow.sklearn.KNNWorkflow", 55, 90, 4)
    ])
    def test_that_preprocessors_are_logged_in_output(
            self,
            openmlid,
            workflow,
            min_num_cols_expected_after_first_step,
            max_num_cols_expected_after_first_step,
            num_cols_expected_after_last_step):

        from lcdb.builder.utils import import_attr_from_module

        WorkflowClass = import_attr_from_module(workflow)
        config_space = WorkflowClass.config_space()
        config = dict(config_space.get_default_configuration())

        config.update({
            "pp@cat_encoder": "onehot",
            "pp@decomposition": "kernel_pca",
            "pp@featuregen": "poly",
            "pp@featureselector": "selectp",
            "pp@scaler": "minmax",
            "pp@kernel_pca_kernel": "linear",
            "pp@kernel_pca_n_components": 0.25,
            "pp@poly_degree": 2,
            "pp@selectp_percentile": 25,
            "pp@std_with_std": True
        })

        output = run_learning_workflow(
            openml_id=openmlid,
            workflow_class=workflow,
            workflow_parameters=config,
            task_type="classification",
            monotonic=False,
            valid_seed=0,
            test_seed=0,
            workflow_seed=0,
            valid_prop=0.1,
            test_prop=0.1,
            timeout_on_fit=60,
            anchor_schedule="power",
            epoch_schedule="power-2-2-2",
            raise_errors=True,
            logger=logger
        )

        metadata = QueryDatasetMetadata()(output["metadata"]["json"])
        init_cols = metadata["cols"]
        anchors = QueryAnchorValues()(output["metadata"]["json"])

        pp_results_per_fold_and_anchor = [
                QueryPreprocessorResults(fold=key)(output["metadata"]["json"])
                for key in ["train", "valid", "test"]
            ]

        self.assertEqual(3, len(pp_results_per_fold_and_anchor))
        for applications_per_anchor in pp_results_per_fold_and_anchor:
            self.assertEqual(len(anchors), len(applications_per_anchor))
            for application_in_anchor in applications_per_anchor:
                self.assertEqual(5, len(application_in_anchor))
        pp_matrix = np.array(pp_results_per_fold_and_anchor).transpose(1, 0, 2)
        self.assertEqual((len(anchors), 3, 5), pp_matrix.shape)

        for anchor_index, (anchor, pp_results_of_anchor) in enumerate(zip(anchors, pp_matrix)):

            for i in range(5):

                # check consistency (in pp names and dimensionality) across train, validation, and test folds
                self.assertEqual(pp_results_of_anchor[0, i]["tag"], pp_results_of_anchor[1, i]["tag"])  # same pp applied to train and validation
                self.assertEqual(pp_results_of_anchor[0, i]["tag"], pp_results_of_anchor[2, i]["tag"])  # same pp applied to train and test

                self.assertEqual(pp_results_of_anchor[0, i]["metadata"]["new_shape"]["cols"],
                                 pp_results_of_anchor[1, i]["metadata"]["new_shape"]["cols"])  # same dimensionality in train and validation
                self.assertEqual(pp_results_of_anchor[0, i]["metadata"]["new_shape"]["cols"],
                                 pp_results_of_anchor[1, i]["metadata"]["new_shape"]["cols"])  # same dimensionality in train and test

                # due to consistency, we only get the information of the train fold
                pp_name = pp_results_of_anchor[0, i]["tag"]
                cols_after_pp = pp_results_of_anchor[0, i]["metadata"]["new_shape"]["cols"]
                train_rows_after_pp_train = pp_results_of_anchor[0, i]["metadata"]["new_shape"]["rows"]

                # after numeric pre-processing, the shape should not have changed
                if i == 0:
                    self.assertEqual("pre_numeric_pp", pp_name)
                    self.assertLessEqual(min_num_cols_expected_after_first_step, cols_after_pp)
                    if cols_after_pp < init_cols:
                        logger.warning(
                            f"Initial preprocessor has dropped {init_cols - cols_after_pp}"
                            f" columns at anchor {anchor}. This could be because there are no distinct values in those."
                        )
                    self.assertGreaterEqual(max_num_cols_expected_after_first_step, cols_after_pp)
                self.assertEqual(anchor, train_rows_after_pp_train)

                if i == 4:  # after last step
                    self.assertEqual(num_cols_expected_after_last_step, cols_after_pp)

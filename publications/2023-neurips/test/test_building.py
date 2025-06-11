import logging

from parameterized import parameterized
import unittest

from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from lcdb.workflow.xgboost import XGBoostWorkflow
from lcdb.workflow.keras import DenseNNWorkflow
from lcdb.workflow.sklearn import TreesEnsembleWorkflow
from lcdb.builder import run_learning_workflow
from lcdb.builder.utils import import_attr_from_module
import itertools as it
import json

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
    41138
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
    "lcdb.workflow.xgboost.XGBoostWorkflow",
    "lcdb.workflow.keras.DenseNNWorkflow"
]

VAL_SEEDS = [0]
TEST_SEEDS = [0]
WORKFLOW_SEEDS = [0]

MAX_SAMPLE_ANCHOR = 512


class TestBuildFunctionalities(unittest.TestCase):

    @parameterized.expand(list(it.product([61], WORKFLOWS, VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS, [True, False])))
    def test_workflow_base_functionality_and_integrity(self, openmlid, workflow, val_seed, test_seed, workflow_seed, monotonic):

        workflow_class = import_attr_from_module(workflow)

        params = {}
        if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
            params["pp@cat_encoder"] = "onehot"
            
        if issubclass(workflow_class, XGBoostWorkflow):
            params["n_estimators"] = 16
        
        if issubclass(workflow_class, TreesEnsembleWorkflow):
            params["n_estimators"] = 16
        
        if issubclass(workflow_class, DenseNNWorkflow):
            params["epoch_schedule"] = "linear"
            params["num_epochs"] = 10

        logger.info(f"Starting test of workflow {workflow} on dataset {openmlid}")
        try:
            out = run_learning_workflow(
                openml_id=openmlid,
                workflow_class=workflow,
                workflow_parameters=params,
                valid_seed=val_seed,
                test_seed=test_seed,
                workflow_seed=workflow_seed,
                monotonic=monotonic,
                raise_errors=True,
                anchor_schedule="power-2-2-2",
                max_sample_anchor=MAX_SAMPLE_ANCHOR
            )

            parsed_json = json.loads(out["metadata"]["json"])

            final_node = parsed_json["children"][-1]
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

            test_timestamp_consistency(parsed_json)

        except Exception as e:
            msg = str(e)
            if "covariance is ill defined" in msg:
                pass
            else:
                raise e

    @parameterized.expand(list(it.product(DATASETS, WORKFLOWS, VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS, [True, False])))
    def test_ability_to_work_all_types_of_datasets(self, openmlid, workflow, val_seed, test_seed, workflow_seed, monotonic):

        workflow_class = import_attr_from_module(workflow)

        params = {}
        if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
            params["pp@cat_encoder"] = "onehot"
            
        if issubclass(workflow_class, XGBoostWorkflow):
            params["n_estimators"] = 16

        if issubclass(workflow_class, TreesEnsembleWorkflow):
            params["n_estimators"] = 16
        
        if issubclass(workflow_class, DenseNNWorkflow):
            params["epoch_schedule"] = "linear"
            params["num_epochs"] = 10

        logger.info(f"Starting test of workflow {workflow} on dataset {openmlid}")
        try:
            out = run_learning_workflow(
                openml_id=openmlid,
                workflow_class=workflow,
                workflow_parameters=params,
                valid_seed=val_seed,
                test_seed=test_seed,
                workflow_seed=workflow_seed,
                monotonic=monotonic,
                raise_errors=True,
                anchor_schedule="power-2-2-2",
                max_sample_anchor=MAX_SAMPLE_ANCHOR
            )

            parsed_json = json.loads(out["metadata"]["json"])

            final_node = parsed_json["children"][-1]
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

            test_timestamp_consistency(parsed_json)

        except Exception as e:
            msg = str(e)
            if "covariance is ill defined" in msg:
                pass
            else:
                raise e

    @parameterized.expand([
        (3, "lcdb.workflow.sklearn.KNNWorkflow", 40),
        (188, "lcdb.workflow.sklearn.KNNWorkflow", 90)
    ])
    def test_that_preprocessors_are_logged_in_output(
            self,
            openmlid,
            workflow,
            max_num_cols_expected_after_first_step):

        from lcdb.builder.utils import import_attr_from_module

        WorkflowClass = import_attr_from_module(workflow)
        config_space = WorkflowClass.config_space()
        config = dict(config_space.get_default_configuration())

        portion_retained_in_feature_selection = 0.3

        config.update({
            "pp@cat_encoder": "onehot",
            "pp@decomposition": "kernel_pca",
            "pp@featuregen": "poly",
            "pp@featureselector": "selectp",
            "pp@scaler": "minmax",
            "pp@kernel_pca_kernel": "linear",
            "pp@kernel_pca_n_components": 0.25,
            "pp@poly_degree": 2,
            "pp@selectp_percentile": int(100 * portion_retained_in_feature_selection),
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

        parsed_json = json.loads(output["metadata"]["json"])

        metadata = QueryDatasetMetadata()(parsed_json)
        init_cols = metadata["cols"]
        anchors = QueryAnchorValues()(parsed_json)

        pp_results_per_fold_and_anchor = [
                QueryPreprocessorResults(fold=key)(parsed_json)
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
            
            num_cols_received_from_previous_step = None

            for i in range(5):  # go over the five pre-processing steps

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

                # check that the right number of rows were used for training
                self.assertEqual(anchor, train_rows_after_pp_train)

                # after numeric pre-processing, the shape should not have changed
                if i == 0:
                    self.assertEqual("pre_numeric_pp", pp_name)
                    self.assertLessEqual(5, cols_after_pp)  # we should have at least 5 features after init step
                    if cols_after_pp < init_cols:
                        logger.warning(
                            f"Initial preprocessor has dropped {init_cols - cols_after_pp}"
                            f" columns at anchor {anchor}. This could be because there are no distinct values in those."
                        )
                    self.assertGreaterEqual(max_num_cols_expected_after_first_step, cols_after_pp)
                    num_features_accepted = [cols_after_pp]  # disable the check here since we cannot know the exact value
                    
                elif i == 1: # feature selection
                    num_features_accepted = [
                        int(np.floor(num_cols_received_from_previous_step * portion_retained_in_feature_selection)),
                        int(np.ceil(num_cols_received_from_previous_step * portion_retained_in_feature_selection))
                    ] # we select 25% of the features
                    
                    # this is only used for the PCA since the number of components are configured based on this
                    num_features_expected_after_feature_selection = np.round(num_cols_received_from_previous_step * portion_retained_in_feature_selection)
                
                elif i == 2: # feature generation
                    num_features_accepted = [2 * num_cols_received_from_previous_step + (num_cols_received_from_previous_step * (num_cols_received_from_previous_step - 1)) // 2]
                    num_features_expected_after_generation = 2 * num_features_expected_after_feature_selection + (num_features_expected_after_feature_selection * (num_features_expected_after_feature_selection - 1)) // 2
                
                elif i == 3: # feature scaling
                    num_features_accepted = [num_cols_received_from_previous_step]
                elif i == 4:  # PCA
                    num_features_accepted = [
                        min(anchor, int(np.floor(num_features_expected_after_generation / 4))),
                        min(anchor, int(np.ceil(num_features_expected_after_generation / 4)))
                    ] # we project to 25% of the features
                
                self.assertIn(
                    cols_after_pp,
                    num_features_accepted,
                    f"Expected {num_features_accepted} features after pre-processing step {i + 1} at anchor {anchor} but found {cols_after_pp}. "
                    f"Received {num_cols_received_from_previous_step} features from previous step."
                ) 
                
                num_cols_received_from_previous_step = cols_after_pp

    @parameterized.expand([
        (1111, "lcdb.workflow.sklearn.LibLinearWorkflow", 0, 0, 0, 16),
        (1457, "lcdb.workflow.sklearn.LibLinearWorkflow", 0, 0, 0, 16),
        (1457, "lcdb.workflow.sklearn.KNNWorkflow", 0, 0, 0, 16)
    ])
    def test_correct_behavior_on_degenerated_anchors(self, openmlid, workflow, val_seed, test_seed, workflow_seed, anchor):

        for monotonic in [False, True]:
            workflow_class = import_attr_from_module(workflow)

            if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
                params = {
                    "pp@cat_encoder": "onehot"
                }
            else:
                params = None

            logger.info(f"Starting test of workflow {workflow} on dataset {openmlid}")
            out = run_learning_workflow(
                openml_id=openmlid,
                workflow_class=workflow,
                workflow_parameters=params,
                valid_seed=val_seed,
                test_seed=test_seed,
                workflow_seed=workflow_seed,
                raise_errors=True,
                monotonic=monotonic,
                anchor_schedule=str(anchor),
                epoch_schedule="power-2-2-2"
            )

            parsed_json = json.loads(out["metadata"]["json"])

            final_node = parsed_json["children"][-1]
            self.assertEqual("build_curves", final_node["tag"])
            first_anchor_in_final_node = final_node["children"][0]
            self.assertEqual("anchor", first_anchor_in_final_node["tag"])
            self.assertEqual(anchor, first_anchor_in_final_node["metadata"]["value"])
            metrics_in_first_anchor_in_final_node = first_anchor_in_final_node["children"][-1]
            self.assertEqual("metrics", metrics_in_first_anchor_in_final_node["tag"])
            validation_confusion_matrix_in_first_anchor_in_final_node = metrics_in_first_anchor_in_final_node["children"][1]["children"][0]
            self.assertEqual("confusion_matrix", validation_confusion_matrix_in_first_anchor_in_final_node["tag"])

    @parameterized.expand(list(it.product([61], VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS, [True, False])))
    def test_reproducibility_of_preprocessors(self, openmlid, val_seed, test_seed, workflow_seed, monotonic):

        logger.info(f"Starting reproducibility test of preprocessors on dataset {openmlid}")
        
        from lcdb.workflow.sklearn import DTWorkflow
        from ConfigSpace.hyperparameters import CategoricalHyperparameter
        workflow_class = DTWorkflow

        for hp_name, hp_obj in workflow_class.config_space().items():
            if not hp_name.startswith("pp@") or not isinstance(hp_obj, CategoricalHyperparameter):
                continue

            for choice in hp_obj.choices:
            
                params = {
                    hp_name: choice
                }
                if hp_name != "pp@cat_encoder" and openmlid in [3, 188]:
                    params["pp@cat_encoder"] = "onehot"
                    
                matrices_first_run = []

                for run_idx in range(2):
                    out = run_learning_workflow(
                        openml_id=openmlid,
                        workflow_class=workflow_class,
                        workflow_parameters=params,
                        valid_seed=val_seed,
                        test_seed=test_seed,
                        workflow_seed=workflow_seed,
                        monotonic=monotonic,
                        raise_errors=True,
                        anchor_schedule="first",
                        max_sample_anchor=MAX_SAMPLE_ANCHOR,
                        raise_exception_on_unsuitable_preprocessor=False
                    )

                    parsed_json = json.loads(out["metadata"]["json"])
                    final_node = parsed_json["children"][-1]
                    first_anchor_in_final_node = final_node["children"][0]
                    metrics_in_first_anchor_in_final_node = first_anchor_in_final_node["children"][-1]

                    # check that train, validation, and test confusion matrices are identical
                    for i in range(3):
                        matrix = metrics_in_first_anchor_in_final_node["children"][i]["children"][0]["metadata"]["value"]
                        if run_idx == 0:
                            matrices_first_run.append(matrix)
                        else:
                            self.assertEqual(matrices_first_run[i], matrix, msg=f"Missing reproducibility for {workflow_class=}.")


    @parameterized.expand(list(it.product([61], WORKFLOWS, VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS, [True, False])))
    def test_reproducibility_of_actual_workflows(self, openmlid, workflow, val_seed, test_seed, workflow_seed, monotonic):

        workflow_class = import_attr_from_module(workflow)

        params = {}
        if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
            params["pp@cat_encoder"] = "onehot"
            
        if issubclass(workflow_class, XGBoostWorkflow):
            params["n_estimators"] = 16
        
        if issubclass(workflow_class, TreesEnsembleWorkflow):
            params["n_estimators"] = 16
        
        if issubclass(workflow_class, DenseNNWorkflow):
            params["epoch_schedule"] = "linear"
            params["num_layers"] = 2
            params["num_units_first"] = 5
            params["num_units_last"] = 5
            params["num_epochs"] = 5

        logger.info(f"Starting reproducibility test of workflow {workflow} on dataset {openmlid}")
        try:
            
            matrices_first_run = []

            for run_idx in range(2):
                out = run_learning_workflow(
                    openml_id=openmlid,
                    workflow_class=workflow,
                    workflow_parameters=params,
                    valid_seed=val_seed,
                    test_seed=test_seed,
                    workflow_seed=workflow_seed,
                    monotonic=monotonic,
                    raise_errors=True,
                    anchor_schedule="first",
                    max_sample_anchor=MAX_SAMPLE_ANCHOR
                )

                parsed_json = json.loads(out["metadata"]["json"])
                final_node = parsed_json["children"][-1]
                first_anchor_in_final_node = final_node["children"][0]
                metrics_in_first_anchor_in_final_node = first_anchor_in_final_node["children"][-1]

                # check that train, validation, and test confusion matrices are identical
                for i in range(3):
                    matrix = metrics_in_first_anchor_in_final_node["children"][i]["children"][0]["metadata"]["value"]
                    if run_idx == 0:
                        matrices_first_run.append(matrix)
                    else:
                        self.assertEqual(matrices_first_run[i], matrix, msg=f"Missing reproducibility for {workflow_class=}.")
            
        except Exception as e:
            msg = str(e)
            if "covariance is ill defined" in msg:
                pass
            else:
                raise e
import logging

from parameterized import parameterized
import unittest

from ConfigSpace import CategoricalHyperparameter

from lcdb.data._base import load_task
from lcdb.data.split import train_valid_test_split
from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from lcdb.workflow.xgboost import XGBoostWorkflow
from lcdb.workflow.keras import DenseNNWorkflow
from lcdb.workflow.sklearn import SklearnWorkflow, DTWorkflow, TreesEnsembleWorkflow
from lcdb.builder import run_learning_workflow, create_workflow
from lcdb.builder.utils import get_random_state
from lcdb.workflow._util import get_workflow_class, get_config_space_of_workflow
import itertools as it
import json

from lcdb.analysis.json import QueryPreprocessorResults, QueryDatasetMetadata, QueryAnchorValues
import numpy as np

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

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


class TestBuildReproducibility(unittest.TestCase):

    @parameterized.expand(list(it.product(
            [61],
            VAL_SEEDS,
            TEST_SEEDS,
            WORKFLOW_SEEDS,
            [True, False],
            [(hp_name, choice) for hp_name, hp_obj in DTWorkflow.config_space().items() if hp_name.startswith("pp@") and isinstance(hp_obj, CategoricalHyperparameter) for choice in hp_obj.choices]
        )))
    def test_reproducibility_of_preprocessors(self, openmlid, val_seed, test_seed, workflow_seed, monotonic, hyperparameter_setting):

        logger.info(f"Starting reproducibility test of preprocessors on dataset {openmlid}")
        
        workflow_class = DTWorkflow
        hp_name, choice = hyperparameter_setting    
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

    @parameterized.expand(list(it.product(
            [3, 188],
            [
                (workflow_class, hp_name, choice)
                for workflow_class in WORKFLOWS
                for hp_name, hp_obj in get_workflow_class(workflow_class).config_space().items()
                    if hp_name.startswith("pp@") and isinstance(hp_obj, CategoricalHyperparameter)
                for choice in hp_obj.choices
            ],
            WORKFLOW_SEEDS
        )))
    def test_reproducibility_of_basic_workflow_predictions(self, openmlid, parametrized_workflow, workflow_seed):

        (X, y), dataset_metadata = load_task(f"openml.{openmlid}")

        # Transform categorical features
        columns_categories = np.asarray(dataset_metadata["categories"], dtype=bool)
        dataset_metadata["categories"] = {"columns": columns_categories}

        # train/validation/test split
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y, test_seed=0, valid_seed=0)
        X_train = X_train[:256]
        y_train = y_train[:256]

        workflow_class, hp_name, choice = parametrized_workflow
        workflow_class = get_workflow_class(workflow_class)
            
        params = {
            hp_name: choice
        }
        if hp_name == "pp@cat_encoder":
            if openmlid in [3, 188]:
                if choice == "none":
                    return
            else:
                if choice != "none":
                    return

        else:
            if openmlid in [3, 188]:
                params["pp@cat_encoder"] = "onehot"
            else:
                params["pp@cat_encoder"] = "none"

        # make models cheap
        epoch_schedule = "linear"
        if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
            params["pp@cat_encoder"] = "onehot"
            
        if issubclass(workflow_class, XGBoostWorkflow):
            params["n_estimators"] = 4
        
        if issubclass(workflow_class, TreesEnsembleWorkflow):
            params["n_estimators"] = 4
        
        if issubclass(workflow_class, DenseNNWorkflow):
            params["num_layers"] = 2
            params["num_units_first"] = 5
            params["num_units_last"] = 5
            params["num_epochs"] = 5

        for n_jobs in [1, 2]:
            predictions = []
            for _ in range(2):
                workflow = create_workflow(
                    workflow_class=workflow_class,
                    workflow_parameters=params,
                    workflow_seed=workflow_seed,
                    raise_exception_on_unsuitable_preprocessor=True,
                    memory_limit_in_bytes=1024 ** 3,
                    epoch_schedule="first",
                    n_jobs=n_jobs,
                    timer=None,
                    logger=logger
                )

                # now fit the workflow
                with workflow.timer.time("fit"):
                    workflow.fit(X_train, y_train, X_valid, y_valid, X_test, y_test, metadata=dataset_metadata)
                    predictions.append({
                        "y_train_hat_proba": workflow.predict_proba(X_train),
                        "y_valid_hat_proba": workflow.predict_proba(X_valid),
                        "y_test_hat_proba": workflow.predict_proba(X_test)
                    })

            for key in ["y_train_hat_proba", "y_valid_hat_proba", "y_test_hat_proba"]:
                assert np.allclose(predictions[0][key], predictions[1][key]), f"Predictions for {predictions} are not reproducible for workflow {workflow_class} with params {params=}, workflow_seed={workflow_seed}, epoch_schedule={epoch_schedule}, n_jobs={n_jobs}"

    @parameterized.expand(list(it.product([61], WORKFLOWS, VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS, [True])))
    def test_reproducibility_of_outcome_of_full_workflows(self, openmlid, workflow, val_seed, test_seed, workflow_seed, monotonic):

        workflow_class = get_workflow_class(workflow)

        params = {}
        epoch_schedule = "linear"
        if issubclass(workflow_class, PreprocessedWorkflow) and openmlid in [3, 188]:
            params["pp@cat_encoder"] = "onehot"
            
        if issubclass(workflow_class, XGBoostWorkflow):
            params["n_estimators"] = 4
        
        if issubclass(workflow_class, TreesEnsembleWorkflow):
            params["n_estimators"] = 4
        
        if issubclass(workflow_class, DenseNNWorkflow):
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
                    epoch_schedule=epoch_schedule,
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
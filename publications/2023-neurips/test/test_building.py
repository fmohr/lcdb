import logging

from parameterized import parameterized
import unittest

from lcdb.workflow._preprocessing_workflow import PreprocessedWorkflow
from lcdb.builder import run_learning_workflow
from lcdb.builder.utils import import_attr_from_module
import itertools as it

from lcdb.analysis.json import QueryPreprocessorResults

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("LCDB")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.WARN)

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
    "lcdb.workflow.sklearn.TreesEnsembleWorkflow"
]

VAL_SEEDS = [0]
TEST_SEEDS = [0]
WORKFLOW_SEEDS = [0]


class TestBuildFunctionalities(unittest.TestCase):

    @parameterized.expand(list(it.product(DATASETS, WORKFLOWS, VAL_SEEDS, TEST_SEEDS, WORKFLOW_SEEDS)))
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
                epoch_schedule="power"
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
        (3, "lcdb.workflow.sklearn.KNNWorkflow"),
        (188, "lcdb.workflow.sklearn.KNNWorkflow")
    ])
    def test_that_preprocessors_are_logged_in_output(self, openmlid, workflow):

        from lcdb.builder.utils import import_attr_from_module, terminate_on_memory_exceeded

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

        # define stream handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger = logging.getLogger("LCDB")
        logger.addHandler(ch)

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
            logger=logger
        )

        for executed_preprocessors in QueryPreprocessorResults()(output["metadata"]["json"]):
            self.assertEqual(5, len(executed_preprocessors))

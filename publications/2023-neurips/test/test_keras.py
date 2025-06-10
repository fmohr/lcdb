import logging

from parameterized import parameterized
import unittest

from lcdb.workflow.keras import DenseNNWorkflow
from lcdb.workflow.keras._dense import CONFIG_SPACE

from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, IntegerHyperparameter, FloatHyperparameter

import numpy as np
import json

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("LCDB")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

from lcdb.builder import run_learning_workflow

single_change_parametrizations_of_workflow = []
core_parametrization = {}
    
# first add constants that always will be fixed
for hp_name, hp_obj in CONFIG_SPACE.items():
    if isinstance(hp_obj, Constant):
        core_parametrization[hp_name] = hp_obj.value
single_change_parametrizations_of_workflow.append(core_parametrization)

# now check all parametrizations that change exactly one value of the other parameters in comparison to the default
for hp_name, hp_obj in CONFIG_SPACE.items():
    if isinstance(hp_obj, Constant):
        pass
    elif isinstance(hp_obj, CategoricalHyperparameter):
        for val in hp_obj.choices:
            if val != hp_obj.default_value:  # skip default value
                config = core_parametrization.copy()
                config.update({hp_name: val})
                single_change_parametrizations_of_workflow.append(config)
    elif isinstance(hp_obj, IntegerHyperparameter):
        for val in sorted(set([int(v) for v in np.linspace(hp_obj.lower, hp_obj.upper, 4)])):
            if val != hp_obj.default_value:  # skip default value
                config = core_parametrization.copy()
                config.update({hp_name: val})
                single_change_parametrizations_of_workflow.append(config)

    elif isinstance(hp_obj, FloatHyperparameter):
        for val in np.linspace(hp_obj.lower, hp_obj.upper, 4):
            if val != hp_obj.default_value:  # skip default value
                config = core_parametrization.copy()
                config.update({hp_name: val})
                single_change_parametrizations_of_workflow.append(config)
    else:
        raise ValueError(f"Unsupported HP type {type(hp_obj)}")

class TestDenseNetwork(unittest.TestCase):

    @parameterized.expand([(p.copy(), ) for p in single_change_parametrizations_of_workflow])
    def test_functionality_and_reproducibility(self, params):

        logger.info(f"Starting test for configuration {json.dumps(params)}")

        params["epoch_schedule"] = "linear"
        params["num_epochs"] = 3
        if "pp@cat_encoder" not in params:
            params["pp@cat_encoder"] = "onehot"

        openml_id = 3
        anchor_schedule = "first"
        logger.info(
            f"Starting test of params {params} on dataset. To reproduce results, use\n\t"
            f"lcdb test -i {openml_id} -w lcdb.workflow.keras.DenseNNWorkflow --anchor-schedule={anchor_schedule} --parameters='" + json.dumps(params) + "'"
        )

        matrices_first_run = []

        # run the workflow twice to check for reproducibility
        for run_idx in range(2):
            out = run_learning_workflow(
                openml_id=openml_id,
                workflow_class=DenseNNWorkflow,
                workflow_parameters=params,
                valid_seed=0,
                test_seed=0,
                workflow_seed=0,
                monotonic=False,
                raise_errors=True,
                anchor_schedule=anchor_schedule
            )

            # get final node in output
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
                    self.assertEqual(matrices_first_run[i], matrix, msg=f"Missing reproducibility for parametrization {params}.")
import pytest
import unittest

import pandas as pd
import numpy as np
from lcdb.workflow._util import get_config_space_of_workflow
from lcdb.analysis import LearningCurveExtractor, merge_curves
from lcdb import LCDB, Debugger
from parameterized import parameterized
import logging

from lcdb.cli._cli import create_parser

from pathlib import Path

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("LCDB")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


WORKFLOWS = [
    "lcdb.workflow.sklearn.KNNWorkflow",
    "lcdb.workflow.sklearn.LibLinearWorkflow",
    "lcdb.workflow.sklearn.LibSVMWorkflow",
    "lcdb.workflow.sklearn.TreesEnsembleWorkflow",
    "lcdb.workflow.xgboost.XGBoostWorkflow",
    "lcdb.workflow.keras.DenseNNWorkflow"
]


class TestCreation(unittest.TestCase):


    @parameterized.expand(WORKFLOWS)
    def test_basic_creation_with_default(self, workflow):

        logger.info(f"Test lcdb create for -w {workflow}")

        # prepare folder structure
        folder_of_configs = Path(f"{Path(__file__).parent}/run_configs/test_create")
        folder_of_configs.mkdir(exist_ok=True, parents=True)
        config_file = Path(f"{folder_of_configs}/{workflow}.csv")

        # get config space for workflow
        config_space = get_config_space_of_workflow(workflow)

        # create CLI parser
        parser = create_parser()

        # parse command
        num_configs = 10
        args = parser.parse_args(
            f"create"
            f" -w {workflow}"
            f" -n {num_configs}"
            f" -o {config_file}"
        .split())

        # execute command
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)
        
        # check that all configs are there
        df_configs = pd.read_csv(config_file)
        self.assertEqual(num_configs, len(df_configs))

        # check that the first config is the default config
        self.assertEqual(dict(config_space.get_default_configuration()), df_configs.iloc[0].dropna().to_dict())



    @parameterized.expand(WORKFLOWS)
    def test_basic_creation_without_default(self, workflow):

        logger.info(f"Test lcdb create for -w {workflow}")

        # prepare folder structure
        folder_of_configs = Path(f"{Path(__file__).parent}/run_configs/test_create")
        folder_of_configs.mkdir(exist_ok=True, parents=True)
        config_file = Path(f"{folder_of_configs}/{workflow}.csv")

        # get config space for workflow
        config_space = get_config_space_of_workflow(workflow)

        # create CLI parser
        parser = create_parser()

        # parse command
        num_configs = 10
        args = parser.parse_args(
            f"create"
            f" -w {workflow}"
            f" -n {num_configs}"
            f" -o {config_file}"
            f" -ed"
        .split())

        # execute command
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)
        
        # check that all configs are there
        df_configs = pd.read_csv(config_file)
        self.assertEqual(num_configs, len(df_configs))

        # check that none of the configs is the default configs
        default_config = dict(config_space.get_default_configuration())
        for _, config in df_configs.iterrows():
            self.assertNotEqual(default_config, config.dropna().to_dict())

    @parameterized.expand(WORKFLOWS)
    def test_that_a_fraction_of_configs_can_be_kept_default(self, workflow):

        logger.info(f"Test lcdb create for -w {workflow}")

        # prepare folder structure
        folder_of_configs = Path(f"{Path(__file__).parent}/run_configs/test_create")
        folder_of_configs.mkdir(exist_ok=True, parents=True)
        config_file = Path(f"{folder_of_configs}/{workflow}.csv")

        # create CLI parser
        parser = create_parser()

        # parse command
        num_configs = 21
        num_configs_default_learner = num_configs // 3
        num_configs_default_preprocessor = num_configs // 3
        args = parser.parse_args(
            f"create"
            f" -w {workflow}"
            f" -n {num_configs}"
            f" -ndl {num_configs_default_learner}"
            f" -ndp {num_configs_default_preprocessor}"
            f" -ed" # exclude default
            f" -o {config_file}"
        .split())
        

        # execute command
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)
        
        # check that all configs are there
        df_configs = pd.read_csv(config_file)
        self.assertEqual(num_configs, len(df_configs))

        # get config space for workflow
        config_space = get_config_space_of_workflow(workflow)
        default_config = dict(config_space.get_default_configuration())
        default_config_pp = {k: v for k, v in default_config.items() if k.startswith("pp@")}
        default_config_nopp = {k: v for k, v in default_config.items() if not k.startswith("pp@")}

        # mark every config with respect to the defaultness
        df_configs["default_pp"] = df_configs.apply(lambda row: all([row[k] == v for k, v in default_config_pp.items()]), axis=1)
        df_configs["default_learner"] = df_configs.apply(lambda row: all([row[k] == v for k, v in default_config_nopp.items()]), axis=1)
        df_configs["default"] = df_configs["default_pp"] & df_configs["default_learner"]
        self.assertEqual(num_configs_default_learner, df_configs["default_learner"].sum())
        self.assertEqual(num_configs_default_preprocessor, df_configs["default_pp"].sum())
        
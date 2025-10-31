import unittest

import pandas as pd
from lcdb.workflow._util import get_config_space_of_workflow
from parameterized import parameterized
import logging
import numpy as np

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

def check_no_redundancy(df_configs):
    sentinel = object()
    df_filled = df_configs.fillna(sentinel)
    assert not df_filled.duplicated(keep=False).any(), "There are duplicate configs!"


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
    def test_no_redundancy(self, workflow):

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
        num_configs = 10**4
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
        logger.info(f"lcdb create execution finished. Now reading in experiments and checking that we have no redundancy.")

        # check that all configs are there
        df_configs = pd.read_csv(config_file)
        self.assertEqual(num_configs, len(df_configs))

        # check that the first config is the default config
        self.assertEqual(dict(config_space.get_default_configuration()), df_configs.iloc[0].dropna().to_dict())

        # check that there are no redundant entries
        check_no_redundancy(df_configs)

    @parameterized.expand(WORKFLOWS)
    def test_reproducibility(self, workflow):

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
            f" -s 1"
        .split())

        # extract command
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")

        # memorize configs
        config_dfs = []
        for _ in range(2):
            
            # execute command
            print(f"EXECUTING {func}({kwargs})")
            func(**kwargs)

            # check that all configs are there
            df_configs = pd.read_csv(config_file)
            self.assertEqual(num_configs, len(df_configs))
            config_dfs.append(df_configs)

            # check that the first config is the default config
            self.assertEqual(dict(config_space.get_default_configuration()), df_configs.iloc[0].dropna().to_dict())
        
        # check equality of dataframes
        self.assertTrue(pd.DataFrame.equals(config_dfs[0], config_dfs[1]), "config DataFrame not reproducible")


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
        print(df_configs)

        # mark every config with respect to the defaultness
        df_configs["default_pp"] = df_configs.apply(lambda row: all([row[k] == v for k, v in default_config_pp.items()]), axis=1)
        df_configs["default_learner"] = df_configs.apply(lambda row: all([row[k] == v for k, v in default_config_nopp.items()]), axis=1)
        df_configs["default"] = df_configs["default_pp"] & df_configs["default_learner"]
        self.assertEqual(num_configs_default_learner, df_configs["default_learner"].sum())
        self.assertEqual(num_configs_default_preprocessor, df_configs["default_pp"].sum())
        
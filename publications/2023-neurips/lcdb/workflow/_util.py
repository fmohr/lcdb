import ConfigSpace.read_and_write.json
from ConfigSpace import ConfigurationSpace
import json

import os, logging

from func_timeout import func_set_timeout, FunctionTimedOut, func_timeout

from ..data._split import get_splits_for_anchor, get_mandatory_preprocessing
from ..data._openml import get_openml_dataset
from ._base_workflow import BaseWorkflow
from time import time

import itertools as it

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.experimenter import utils

import numpy as np
import openml
import pandas as pd

import sklearn.metrics
from sklearn.pipeline import Pipeline

def get_schedule_for_number_of_instances(num_instances, val_fold_size, test_fold_size):
    max_training_set_size = int((1 - test_fold_size) * (1 - val_fold_size) * num_instances)
    anchors_for_dataset = []
    #for exp in list(range(8, 2 * (1 + int(np.log2(max_training_set_size))) - 1)):
    #    anchors_for_dataset.append(int(np.round(2 ** (exp / 2))))

    # sample size = 2^((7+k)/2) = 2^exponent
    max_k = int(2*np.log2(max_training_set_size) - 7)
    for k in list(range(1, max_k+1)):
        exponent = (7+k)/2
        sample_size = int(np.round(2 ** exponent))
        anchors_for_dataset.append(sample_size)
    if sample_size != max_training_set_size:
        anchors_for_dataset.append(max_training_set_size)
    return anchors_for_dataset

def get_experimenter(learner_class, executor_name="", config_folder="config") -> PyExperimenter:
    config_file = f"{config_folder}/{learner_class.__name__}.cfg"
    return PyExperimenter(
        experiment_configuration_file_path=config_file, name=executor_name
    )

def get_technical_experiment_grid(config_file="config/experiments.cfg", val_fold_size=0.1, test_fold_size=0.1):
    config = utils.load_config(path=config_file)

    # get all combinations for the keyfields that are independent of the dataset properties and the learner
    keyfield_domains = utils.get_keyfield_data(config=config)
    relevant_keyfield_names = ["seed_outer", "seed_inner", "monotonic"]
    keyfield_combinations = list(it.product(*[keyfield_domains[kf] for kf in relevant_keyfield_names]))
    rows = []
    for openmlid in keyfield_domains["openmlid"]:
        num_instances = openml.datasets.get_dataset(openmlid).qualities["NumberOfInstances"]
        schedule = get_schedule_for_number_of_instances(num_instances, val_fold_size, test_fold_size)
        # TODO: Split schedule further and randomize over different jobs (to avoid very long jobs)
        for combo in keyfield_combinations:
            combo = list(combo)
            rows.append([openmlid] + combo[:2] + [schedule] + combo[-1:])
    return pd.DataFrame(rows, columns=["openmlid"] + relevant_keyfield_names[:2] + ["train_sizes"] + relevant_keyfield_names[-1:])

def get_latin_hypercube_sampling(config_space: ConfigurationSpace, num_configs, segmentation=None):
    # TODO: Implement latin hypercube sampling
    # TODO: Make this reproducible
    return config_space.sample_configuration(num_configs)

def unserialize_config_space(json_filename) -> ConfigSpace.ConfigurationSpace:
    with open(json_filename, 'r') as f:
        json_string = f.read()
        return ConfigSpace.read_and_write.json.read(json_string)

def get_all_experiments(workflow_class: BaseWorkflow, num_configs: int, seed: int):

    # get the experiment grid except the hyperparameters
    df_experiments = get_technical_experiment_grid()
    config_space = workflow_class.get_config_space()
    config_space.seed(seed)
    hp_samples = get_latin_hypercube_sampling(config_space=config_space, num_configs=num_configs)

    # create all rows for the experiments
    return [
        {
            "openmlid": openmlid,
            "seed_outer": s_o,
            "seed_inner": s_i,
            "train_sizes": train_sizes,
            "hyperparameters": dict(hp) ,
            "monotonic": mon
        } for (openmlid, s_o, s_i, train_sizes, mon), hp in it.product(df_experiments.values, hp_samples)
    ]

def run(
        openmlid: int,
        workflow_class: BaseWorkflow,
        hyperparameters: dict,
        outer_seed: int,
        inner_seed: int,
        anchors: list,
        monotonic: bool,
        logger=None
):

    if logger is None:
        logger = logging.getLogger("lcdb.exp")

    if type(anchors) != list:
        anchor = [anchors]
    logger.info(f"Starting experiment on {openmlid} for workflow '{workflow_class.__name__}'. Seeds: {outer_seed}/{inner_seed}. Anchors: {anchors}, HPs: {hyperparameters}. Monotonic: {monotonic}")

    # CPU
    logger.info("CPU Settings:")
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        logger.info(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")

    # load data
    binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
    drop_first = False # openmlid not in [3] # drop first cannot be used in datasets with some very rare categorical values
    logger.info(f"Reading dataset. Will be binarized sparsely: {binarize_sparse}")
    X, y = get_openml_dataset(openmlid)
    y = np.array([str(e) for e in y]) # make sure that labels are strings
    logger.info(f"ready. Dataset shape is {X.shape}, label column shape is {y.shape}. Now running the algorithm")
    if X.shape[0] <= 0:
        raise Exception("Dataset size invalid!")
    if X.shape[0] != len(y):
        raise Exception("X and y do not have the same size.")

    results = {}
    for anchor in anchors:
        X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed, monotonic)
        # create the configured workflow
        # TODO: alternatively, one could be lazy and not pass the training data here.
        #       Then the workflow might have to do some setup routine at the beginning of `fit`
        #       Or as a middle-ground solution: We pass the dimensionalities of the task but not the data itself
        logger.info(f"Working on anchor {anchor}, trainset size is {y_train.shape}.")
        workflow = workflow_class(X_train, y_train, hyperparameters)
        try:
            results[anchor] = func_timeout(60*60*2, run_on_data, args=(X_train, X_valid, X_test, y_train, y_valid, y_test, binarize_sparse, drop_first, workflow, logger))
        except KeyboardInterrupt:
            raise
        except FunctionTimedOut:
            results[anchor] = "Timed out"
        except Exception as err:
            results[anchor] = err
    return results

def run_on_data(X_train, X_valid, X_test, y_train, y_valid, y_test, binarize_sparse, drop_first, workflow, logger):
    # get the data for this experiment
    labels = sorted(set(np.unique(y_train)) | set(np.unique(y_valid)) | set(np.unique(y_test)))

    preprocessing_steps = get_mandatory_preprocessing(X_train, y_train, binarize_sparse=binarize_sparse, drop_first=drop_first)
    if preprocessing_steps:
        pl = Pipeline(preprocessing_steps).fit(X_train, y_train)
        X_train, X_valid, X_test = pl.transform(X_train), pl.transform(X_valid), pl.transform(X_test)

    # train the workflow
    logger.debug(f"Start fitting the workflow...")
    ts_fit_start = time()
    workflow.fit((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
    ts_fit_end = time()
    fit_time = ts_fit_end - ts_fit_start

    logger.debug(f"Workflow fitted after {np.round(fit_time, 2)}s. Now obtaining predictions.")

    # compute confusion matrices
    start = time()
    y_hat_train = workflow.predict(X_train)
    predict_time_train = time() - start
    start = time()
    y_hat_valid = workflow.predict(X_valid)
    predict_time_valid = time() - start
    start = time()
    y_hat_test = workflow.predict(X_test)
    predict_time_test = time() - start
    cm_train = sklearn.metrics.confusion_matrix(y_train, y_hat_train, labels=labels)
    cm_valid = sklearn.metrics.confusion_matrix(y_valid, y_hat_valid, labels=labels)
    cm_test = sklearn.metrics.confusion_matrix(y_test, y_hat_test, labels=labels)

    n_train = X_train.shape[0]
    n_valid = X_valid.shape[0]
    n_test = X_test.shape[0]

    n_valid_sub = int(np.ceil(n_train/0.8*0.1))
    n_test_sub = int(np.ceil(n_train/0.8*0.1))
    if n_valid_sub > n_valid:
        n_valid_sub = n_valid
    if n_test_sub > n_test:
        n_test_sub = n_test

    cm_valid_sub = sklearn.metrics.confusion_matrix(y_valid[:n_valid_sub], y_hat_valid[:n_valid_sub], labels=labels)
    cm_test_sub = sklearn.metrics.confusion_matrix(y_test[:n_test_sub], y_hat_test[:n_test_sub], labels=labels)

    # ask workflow to update its summary information (post-processing hook)
    logger.debug("Confusion matrices computed. Computing post-hoc data.")
    workflow.update_summary()

    logger.info("Computation ready, returning results.")
    return labels, cm_train, cm_valid, cm_test, cm_valid_sub, cm_test_sub, fit_time, predict_time_train, predict_time_valid, predict_time_test, workflow.summary
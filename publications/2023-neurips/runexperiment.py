from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
import logging

from evalutils import *

import json

import sys


    
logger = logging.getLogger("exp")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_logger = logging.getLogger("evalutils")
eval_logger.setLevel(logging.DEBUG)

def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_config):
    
    print("Executing")
    
    # Extracting given parameters
    openmlid = int(keyfields['openmlid'])
    outer_seed = int(keyfields['seed_outer'])
    inner_seed = int(keyfields['seed_inner'])
    train_size = int(keyfields['train_size'])
    hps = json.loads(keyfields['hyperparameters'])
    c = hps["c"]
    gamma = hps["gamma"]
    monotonic = bool(keyfields['monotonic'])
    
    
    logger.info(f"Starting experiment for openmlid {openmlid} with {train_size} training instances and seeds {outer_seed}/{inner_seed}. c = {c}, gamma = {gamma}. Monotonic folds: {monotonic}")
    
    # treat data sparse?
    binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
    labels, cms_train, cms_valid, cms_test, traintimes, predict_times_train, predict_times_valid, predict_times_test = compute_curve_for_svc(openmlid, outer_seed, inner_seed, train_size, monotonic, c, gamma, logger = logger)
    
    # Write intermediate results to database
    out = {
        "labels": [str(v) for v in labels],
        "cm_train": [[int(v) for v in a.flatten()] for a in cms_train],
        "cm_val":[[int(v) for v in a.flatten()] for a in cms_valid],
        "cm_test": [[int(v) for v in a.flatten()] for a in cms_test],
        "time_train": np.round(traintimes, 8),
        "time_predict_train": np.round(predict_times_train, 8),
        "time_predict_val": np.round(predict_times_valid, 8),
        "time_predict_test": np.round(predict_times_test, 8)
    }
    resultfields = {
        'result': str(out)
    }
    result_processor.process_results(resultfields)
    
    logger.info("Finished")
    

if __name__ == '__main__':
    job_name = sys.argv[1]
    experimenter = PyExperimenter(
        #database_credential_file_path = 'config/database_credentials.cfg',
        experiment_configuration_file_path="config/experiments_svm.cfg",
        name=job_name
    )
    print("Starting")
    experimenter.execute(run_experiment, max_experiments=-1, random_order=True)
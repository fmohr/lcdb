from py_experimenter.experimenter import PyExperimenter
import itertools as it
import numpy as np
import json
import openml
from tqdm import tqdm

# get experimenter object
experiment_configuration_file_path = 'config/experiments_svm.cfg'
experimenter = PyExperimenter(
    experiment_configuration_file_path=experiment_configuration_file_path, name='lcdb2'
)

# define domains for experiments
granularity = 2
domain_datasets = [3, 6]

# create list of all datasets with all possible training set sizes on that dataset
print(f"Summary of training set sizes for the different datasets (by openmlids):")
domain_datasets_with_sizes = []
test_fold_size = 0.1
val_fold_size = 0.1
for openmlid in domain_datasets:
    num_instances = openml.datasets.get_dataset(openmlid).qualities["NumberOfInstances"]
    anchors_for_dataset = []
    for exp in list(range(4, int(np.log2(num_instances)))):
        anchors_for_dataset.append(int(np.round(2**exp)))
    anchors_for_dataset.append(int((1 - test_fold_size) * (1 - val_fold_size) * num_instances))
    
    print(f"\t{openmlid}: {anchors_for_dataset}")
    for a in anchors_for_dataset:
        domain_datasets_with_sizes.append((openmlid, a))

# set "simple" experiment parameters
domain_seed_outer = [0]
domain_seed_inner = [0]
domain_monotonic = [True, False]

# configure hyperparameters
domain_hyperparameter_c = [np.round(10**i, 5) for i in np.linspace(-5, 5, granularity)]
domain_hyperparameter_gamma = [np.round(10**i, 5) for i in np.linspace(-5, 5, granularity)]
domain_hyperparameters = [json.dumps({"c": c, "gamma": gamma}) for c, gamma in it.product(domain_hyperparameter_c, domain_hyperparameter_gamma)]

# create all rows for the experiments
experimenter.fill_table_with_rows(rows=[
    {
        "openmlid": openmlid,
        "seed_outer": s_o,
        "seed_inner": s_i,
        "train_size": train_size,
        "hyperparameters": hp,
        "monotonic": mon
    } for ((openmlid, train_size), s_o, s_i, hp, mon) in it.product(
        domain_datasets_with_sizes,
        domain_seed_outer,
        domain_seed_inner,
        domain_hyperparameters,
        domain_monotonic 
    )
])
"""Command line to create/generate new experiments."""


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "create"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Create new experiments from a configuration file."
    )

    subparser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    subparser.set_defaults(func=function_to_call)


def main(config: str, *args, **kwargs):
    """
    :meta private:
    """
    
    import itertools as it
    import json

    import numpy as np
    import openml
    from py_experimenter.experimenter import PyExperimenter

    # get experimenter object
    experiment_configuration_file_path = 'config/experiments_svm.cfg'
    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path, name='lcdb2'
    )

    # define domains for experiments
    granularity = 40
    domain_datasets = [3, 6]

    # create list of all datasets with all possible training set sizes on that dataset
    print(f"Summary of training set sizes for the different datasets (by openmlids):")
    domain_datasets_with_sizes = []
    test_fold_size = 0.1
    val_fold_size = 0.1
    for openmlid in domain_datasets:
        num_instances = openml.datasets.get_dataset(openmlid).qualities["NumberOfInstances"]
        max_training_set_size = int((1 - test_fold_size) * (1 - val_fold_size) * num_instances)
        anchors_for_dataset = []
        for exp in list(range(4, int(np.log2(max_training_set_size)))):
            anchors_for_dataset.append(int(np.round(2**exp)))
        anchors_for_dataset.append(max_training_set_size)
        
        print(f"\t{openmlid}: {anchors_for_dataset}")
        for a in anchors_for_dataset:
            domain_datasets_with_sizes.append((openmlid, a))

    # set "simple" experiment parameters
    domain_seed_outer = list(range(5))
    domain_seed_inner = list(range(5))
    domain_monotonic = [True, False]

    # configure hyperparameters
    domain_hyperparameter_c = [np.round(10**i, 5) for i in np.linspace(-10, 10, granularity)]
    domain_hyperparameter_gamma = [np.round(10**i, 5) for i in np.linspace(-10, 10, granularity)]
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

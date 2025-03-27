# Experiments on Snellius

### Installation
To set up the environment and install necessary dependencies, follow these steps:

1. Clone LCDB repository (the branch here is deephyper):

    ```bash
    git clone -b dev git@github.com:fmohr/lcdb.git
    ```
2. Run the installation script:
    ```bash
    cd lcdb/publications/2023-neurips
    ./install/snellius.sh
    ```
3. Experiments execution:
    -  Move to the experiment directory:
    ```bash
    cd experiments/surf/snellius
    ```
    - Give proper access to scripts
    ```bash
    chmod -R u+x scripts/*
    ```
    - To run the experiments, you need to specify the OpenML datases to be executed in `datasets_to_test.csv`.
    - Setup configuration `config.yaml` file. Set desired values:
    ```yaml
    workflow_name: {workflow name: [liblinear,libsvm,knn,treesensemble,xgboost]}
    desired_memory_GB: {memory in GBs per configuration/job up to 336}
    config_num: {number of configurations to test}
    workflow_seed: {workflow seed}
    test_seeds: {list of test seeds}
    val_seeds: {list of validation seeds}
    ```
    Note: the desired memory will be recalculated within the run script by calculating the number of cores to be used and recalculating the memory per core. A notification will be provided and the configuration will be updated in the `config.yaml` file.
    - To run the experiments:
    ```bash
    ./scripts/run_wrapper.sh
    ```
    - To collect results and upload to pcloud, you need to first ensure that the `pcloud_token` is updated (they have an expiration time). To do update this run the notebook `lcdb/publications/2023-neurips/repo_connect.ipynb`. This will update `.env` variables. Then run the following script:
    ```bash
    ./scripts/campaign_wrapper.sh
    ```
    - for analysis, you can use the notebook `lcdb/publications/2023-neurips/analysis.ipynb`




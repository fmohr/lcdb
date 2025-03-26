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
    - Setup configuration `config.yaml` file. Set desired cores to be used and call script to get memory per core
    ```bash
    ./scripts/get_memory_per_core.sh 
    ```
    - To submit the jobs to cluster run the following command with a valid workflow name [`liblinear`,`libsvm`,`knn`,`treesensemble`,`xgboost`]:
    ```bash
    ./scripts/run_wrapper.sh <workflow name>
    ```
    - To collect results to pcloud:
    ```bash
    ./scripts/campaign_wrapper.sh <memory in GBs>
    ```




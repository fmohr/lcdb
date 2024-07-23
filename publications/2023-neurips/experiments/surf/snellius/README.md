# Experiments on Snellius

### Installation
To set up the environment and install necessary dependencies, follow these steps:

1. Clone LCDB repository (the branch here is deephyper):

    ```bash
    git clone -b deephyper git@github.com:fmohr/lcdb.git
    ```
2. Build the environment:
    ```bash
    mkdir -p lcdb/publications/2023-neurips/build && cd $_
    ```
3. Run the installation script:
    ```bash
    ../install/snellius.sh
    ```
4. Activate the DeepHyper environment:
    ```bash
    source activate-dh-env.sh
    ```


### Example

To run the experiments, you need to specify the OpenML dataset in `datasets_to_test.csv`. 

To submit the jobs to cluster: 
```bash
sbatch wrapper_script.sh liblinear
```
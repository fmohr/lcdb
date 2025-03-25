# Experiments on Snellius

### Installation
To set up the environment and install necessary dependencies, follow these steps:

1. Clone LCDB repository (the branch here is deephyper):

    ```bash
    git clone -b dev git@github.com:fmohr/lcdb.git
    ```

2. Run the installation script:
    ```bash
    ./workspace/lcdb/publications/2023-neurips/install/snellius.sh
    ```

4. Activate the DeepHyper environment:
    ```bash
    source activate-dh-env.sh
    ```


### Example

To run the experiments, you need to specify the OpenML dataset in `datasets_to_test.csv`. 

To submit the jobs to cluster: 
```bash
./scripts/wrapper_script.sh <workflow name>
```

To collect results to pcloud:
```bash
./scripts/campaign_wrapper.sh <memory in GBs>
```

The workflow names are:
- `liblinear`
- `libsvm`
- `knn`
- `treesensemble`
- `xgboost`

The memory in GBs is calculated as follows (code from `run_wrapper.sh`):
```bash
# *********MEMORY CALCULATIONS*********
# Number of nodes
NODES=1
CPUS_PER_TASK=192 # number of cores per node (genoa)
export DESIRED_CORES=5 # desired number of cores per task (workflow-dataset-seed triplet)
MEMORY_PER_NODE_GB=336 # memory in Genoa

# Memory calculations
TOTAL_MEMORY_GB=$((MEMORY_PER_NODE_GB * NODES))
TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))
MEMORY_PER_CORE=$((TOTAL_MEMORY_MB / DESIRED_CORES))
export LCDB_WORKFLOW_MEMORY_LIMIT=$MEMORY_PER_CORE
export LCDB_WORKFLOW_MEMORY_LIMIT_GB=$((MEMORY_PER_CORE / 1024))
# *********MEMORY CALCULATIONS*********```
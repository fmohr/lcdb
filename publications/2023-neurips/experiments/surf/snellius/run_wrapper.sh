#!/bin/bash


set -xe

# get the workflow
declare -A mapping
mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
)

# Retrieve the value based on the key
if [[ -n "${mapping[$1]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$1]}
    export LCDB_OUTPUT_WORKFLOW=$PWD/$1/output/$LCDB_WORKFLOW

else
    echo "Invalid algorithm: '$1'"
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Redirect output and error to the appropriate log file
log_dir="$PWD/$1"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper.log") 2>&1

# Define the range of seeds
val_seeds=(0 1 2 3 4)
test_seeds=(0 1 2 3 4)

# create the configs to be executed
source config.sh
./create.sh

# Number of nodes
# TODO: Not yet supported but should investigate
NODES=1

for val_seed in "${val_seeds[@]}"; do
    for test_seed in "${test_seeds[@]}"; do

        # Default setting 128
        CPUS_PER_TASK=32   
        # Total memory per node in GB
        MEMORY_PER_NODE_GB=224
        
        # Calculate total memory across all nodes and convert to MB
        TOTAL_MEMORY_GB=$((MEMORY_PER_NODE_GB * NODES))
        TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))

        initial_memory_per_task_mb=$((TOTAL_MEMORY_MB / CPUS_PER_TASK))
        LCDB_EVALUATION_MEMORY_LIMIT=$((initial_memory_per_task_mb - 128))

        export LCDB_VALID_SEED=$val_seed
        export LCDB_TEST_SEED=$test_seed

        # Get the size of the array of datasets
        array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

        WRAPPER_SCRIPT='run.sh'

        # Jobname is assigned the name of the workflow 
        jobname=''$1''

        # Submit the job using sbatch
        sbatch --export=all --job-name=$jobname --array=0-$array_size --cpus-per-task=$CPUS_PER_TASK --mem-per-cpu=$LCDB_EVALUATION_MEMORY_LIMIT --nodes=$NODES "$WRAPPER_SCRIPT"
    done
done
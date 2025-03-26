#!/bin/bash
# set -xe
source ~/.bashrc
conda activate lcdb

# Workflow mapping
declare -A mapping
mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"  # Ensure this line exists
)

# Load configuration
CONFIG_FILE="scripts/config.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read the workflow name from config.yaml and remove any extra quotes or spaces
WORKFLOW_NAME=$(yq -r '.workflow_name' "$CONFIG_FILE")

# Ensure that the workflow name exists in the mapping
if [[ -n "${mapping[$WORKFLOW_NAME]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$WORKFLOW_NAME]}
    export LCDB_OUTPUT_WORKFLOW=$PWD/$WORKFLOW_NAME/output/$LCDB_WORKFLOW
else
    echo "Invalid workflow name: '$WORKFLOW_NAME'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Read memory and core configuration
DESIRED_MEMORY_GB=$(yq -r '.desired_memory_GB' "$CONFIG_FILE")
val_seeds=($(yq -r '.val_seeds[]' "$CONFIG_FILE"))
test_seeds=($(yq -r '.test_seeds[]' "$CONFIG_FILE"))

# Print the array values
echo "Test seeds: ${test_seeds[@]}"

log_dir="$PWD/$WORKFLOW_NAME"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper.log") 2>&1

# *********MEMORY CALCULATIONS*********
# Number of nodes
NODES=1
CPUS_PER_TASK=192
MEMORY_PER_NODE_GB=336

# Calculate the number of cores based on the desired memory (in GB)
TOTAL_MEMORY_GB=$((MEMORY_PER_NODE_GB * NODES))
TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))

# Fix: Assign the desired number of cores directly based on desired memory
# Calculate the number of cores to use based on the desired memory
DESIRED_CORES=$((TOTAL_MEMORY_GB / DESIRED_MEMORY_GB))

# Ensure that the number of cores does not exceed the available cores (CPUS_PER_TASK)
DESIRED_CORES=$((DESIRED_CORES > CPUS_PER_TASK ? CPUS_PER_TASK : DESIRED_CORES))

# Memory per core based on the desired memory and number of cores
MEMORY_PER_CORE_GB=$((TOTAL_MEMORY_GB / DESIRED_CORES))
MEMORY_PER_CORE_MB=$((MEMORY_PER_CORE_GB * 1024))
export LCDB_WORKFLOW_MEMORY_LIMIT=$MEMORY_PER_CORE_MB
export LCDB_WORKFLOW_MEMORY_LIMIT_GB=$MEMORY_PER_CORE_GB

# Write the memory per core back to the YAML file
# yq ".desired_memory_GB = '$LCDB_WORKFLOW_MEMORY_LIMIT_GB'" scripts/config.yaml  -i -y
echo "Using $DESIRED_CORES cores and $LCDB_WORKFLOW_MEMORY_LIMIT_GB GB of memory per core."

# *********MEMORY CALCULATIONS*********

# Load config for additional settings
source scripts/config.sh

# Submit create.sh as a Slurm job and force next jobs to wait
create_job_id=$(sbatch --export=ALL --parsable scripts/create.sh)
echo "Submitted create.sh with Job ID: $create_job_id"

for val_seed in "${val_seeds[@]}"; do
    for test_seed in "${test_seeds[@]}"; do
        export LCDB_VALID_SEED=$val_seed
        export LCDB_TEST_SEED=$test_seed
        array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

        WRAPPER_SCRIPT='scripts/run.sh'
        jobname="$WORKFLOW_NAME-$LCDB_WORKFLOW_MEMORY_LIMIT_GB"

        # Submit run.sh with dependency on create.sh
        sbatch --export=all --job-name=$jobname \
            --exclusive \
            --dependency=afterok:$create_job_id \
            --array=0-$array_size \
            --ntasks=$DESIRED_CORES \
            --mem=0 \
            --nodes=$NODES \
            --output=logs/out/%x/openml_idx-%a_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}_mem-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}.log \
            --error=logs/err/%x/openml_idx-%a_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}_mem-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}.err \
            "$WRAPPER_SCRIPT"
    done
done

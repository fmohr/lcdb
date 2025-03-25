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
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
)

# workflow selection
if [[ -n "${mapping[$1]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$1]}
    export LCDB_OUTPUT_WORKFLOW=$PWD/$1/output/$LCDB_WORKFLOW
else
    echo "Invalid algorithm: '$1'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

log_dir="$PWD/$1"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper.log") 2>&1

# load  configuration
CONFIG_FILE="scripts/config.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

export DESIRED_CORES=$(yq '.desired_cores' "$CONFIG_FILE")
val_seeds=($(yq '.val_seeds[]' "$CONFIG_FILE"))
test_seeds=($(yq '.test_seeds[]' "$CONFIG_FILE"))

# *********MEMORY CALCULATIONS*********
# Number of nodes
NODES=1
CPUS_PER_TASK=192
MEMORY_PER_NODE_GB=336

# Memory calculations
TOTAL_MEMORY_GB=$((MEMORY_PER_NODE_GB * NODES))
TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))
MEMORY_PER_CORE=$((TOTAL_MEMORY_MB / DESIRED_CORES))
export LCDB_WORKFLOW_MEMORY_LIMIT=$MEMORY_PER_CORE
export LCDB_WORKFLOW_MEMORY_LIMIT_GB=$((MEMORY_PER_CORE / 1024))
# *********MEMORY CALCULATIONS*********
# Load config
source scripts/config.sh

# submit create.sh as a Slurm job and force next jobs to wait
create_job_id=$(sbatch --export=ALL --parsable scripts/create.sh)
echo "Submitted create.sh with Job ID: $create_job_id"


for val_seed in "${val_seeds[@]}"; do
    for test_seed in "${test_seeds[@]}"; do
        export LCDB_VALID_SEED=$val_seed
        export LCDB_TEST_SEED=$test_seed
        array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

        WRAPPER_SCRIPT='scripts/run.sh'
        jobname="$1-$CPUS_PER_TASK"

            # --cpus-per-task=$CPUS_PER_TASK \
            # --mem-per-cpu=$LCDB_EVALUATION_MEMORY_LIMIT \
        # submitting run.sh with dependency on create.sh
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
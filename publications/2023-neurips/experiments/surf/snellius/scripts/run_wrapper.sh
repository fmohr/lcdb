#!/bin/bash
# set -xe

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

# Validate workflow selection
if [[ -n "${mapping[$1]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$1]}
    export LCDB_OUTPUT_WORKFLOW=$PWD/$1/output/$LCDB_WORKFLOW
else
    echo "Invalid algorithm: '$1'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Redirect output and error logs
log_dir="$PWD/$1"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper.log") 2>&1

# Load config
source scripts/config.sh

# Submit create.sh as a Slurm job
create_job_id=$(sbatch --export=ALL --parsable scripts/create.sh)
echo "Submitted create.sh with Job ID: $create_job_id"

val_seeds=(0)
test_seeds=(0 1)

# Number of nodes
NODES=1

for val_seed in "${val_seeds[@]}"; do
    for test_seed in "${test_seeds[@]}"; do
        CPUS_PER_TASK=192
        export DESIRED_CORES=6
        MEMORY_PER_NODE_GB=336
        
        # Memory calculations
        TOTAL_MEMORY_GB=$((MEMORY_PER_NODE_GB * NODES))
        TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))
        MEMORY_PER_CORE=$((TOTAL_MEMORY_MB / DESIRED_CORES))
        export LCDB_WORKFLOW_MEMORY_LIMIT=$MEMORY_PER_CORE

        initial_memory_per_task_mb=$((TOTAL_MEMORY_MB / CPUS_PER_TASK))
        LCDB_EVALUATION_MEMORY_LIMIT=$((initial_memory_per_task_mb - 128))
        echo "Memory per core/config: $LCDB_EVALUATION_MEMORY_LIMIT"
        echo "Total memory: $TOTAL_MEMORY_MB"
        echo "CPUS_PER_TASK: $CPUS_PER_TASK"

        export LCDB_VALID_SEED=$val_seed
        export LCDB_TEST_SEED=$test_seed
        array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

        WRAPPER_SCRIPT='scripts/run.sh'
        jobname="$1-$CPUS_PER_TASK"

        # Submit run.sh with dependency on create.sh
        sbatch --export=all --job-name=$jobname \
            --dependency=afterok:$create_job_id \
            --array=0-$array_size \
            --cpus-per-task=$CPUS_PER_TASK \
            --mem-per-cpu=$LCDB_EVALUATION_MEMORY_LIMIT \
            --nodes=$NODES \
            --output=logs/out/%x/openml_idx-%a_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}_mem-${MEMORY_PER_CORE}.log \
            --error=logs/err/%x/openml_idx-%a_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}_mem-${MEMORY_PER_CORE}.err \
            "$WRAPPER_SCRIPT"
    done
done
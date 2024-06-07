#!/bin/bash


# get the workflow
declare -A mapping
mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
)

# Retrieve the value based on the key
if [[ -n "${mapping[$1]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$1]}
    export LCDB_OUTPUT_WORKFLOW=$PWD/$1/output/$LCDB_WORKFLOW

else
    echo "Invalid algorithm: '$1'"
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Default setting 128
CPUS_PER_TASK=32
# Total memory in GB
TOTAL_MEMORY_GB=224
# Calculate initial memory limit per task in MB (multiply by 1024 to convert GB to MB)
TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))
initial_memory_per_task_mb=$((TOTAL_MEMORY_MB / CPUS_PER_TASK))
LCDB_EVALUATION_MEMORY_LIMIT=$((initial_memory_per_task_mb - 128))

source config.sh
# Move to directory for workflow to be run
# Create the configs to be executed
./create.sh


# Get the size of the array of datasets
array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

WRAPPER_SCRIPT='run.sh'

# moving to the workflow directory for logging
# cd "$1"
# Jobname is assigned the name of the workflow 
jobname=''$1''

# Submit the job using sbatch
sbatch --export=all --job-name=$jobname --array=0-$array_size --cpus-per-task=$CPUS_PER_TASK --mem-per-cpu=$LCDB_EVALUATION_MEMORY_LIMIT "$WRAPPER_SCRIPT"
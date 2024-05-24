#!/bin/bash
# Move to directory for workflow to be run
cd "$1"

# Default setting 128
CPUS_PER_TASK=64
# Total memory in GB
TOTAL_MEMORY_GB=224
# Calculate initial memory limit per task in MB (multiply by 1024 to convert GB to MB)
TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))
initial_memory_per_task_mb=$((TOTAL_MEMORY_MB / CPUS_PER_TASK))
LCDB_EVALUATION_MEMORY_LIMIT=$((initial_memory_per_task_mb - 128))

source config.sh
# Create the configs to be executed
./create.sh

# Get the size of the array of datasets
array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

WRAPPER_SCRIPT='../run.sh'

# Jobname is assigned the name of the workflow 
jobname=''$1''

# Submit the job using sbatch
sbatch --job-name=$jobname --array=0-$array_size --cpus-per-task=$CPUS_PER_TASK --mem-per-cpu=$LCDB_EVALUATION_MEMORY_LIMIT "$WRAPPER_SCRIPT"
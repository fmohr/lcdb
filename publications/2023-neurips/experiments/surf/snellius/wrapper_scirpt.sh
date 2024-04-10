#!/bin/bash
# Move to directory for workflow to be run
cd "$1"

source config.sh
# Create the configs to be executed
./create.sh

# Get the size of the array of datasets
array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

WRAPPER_SCRIPT='run.sh'

# Jobname is assigned the name of the workflow 
jobname=''$1''

# Submit the job using sbatch
sbatch --job-name=$jobname --array=0-$array_size "$WRAPPER_SCRIPT"
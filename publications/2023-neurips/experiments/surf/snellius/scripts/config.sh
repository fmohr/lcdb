#!/bin/bash

# load  configuration
export path_to_snellius="/home/$USER/workspace/lcdb/publications/2023-neurips/experiments/surf/snellius"
CONFIG_FILE="$path_to_snellius/scripts/config.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

export LCDB_NUM_CONFIGS=$(yq '.config_num' "$CONFIG_FILE")
export LCDB_WORKFLOW_SEED=($(yq '.workflow_seed' "$CONFIG_FILE"))

# Read the CSV file and load into an array
declare -a values
while IFS= read -r line || [[ -n "$line" ]]; do
    values+=("$line")
done < "$path_to_snellius/datasets_to_test.csv"
export LCDB_OPENML_ID_ARRAY=(${values[@]})

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
  export LCDB_OPENML_ID=3
else
  export LCDB_OPENML_ID=${LCDB_OPENML_ID_ARRAY[SLURM_ARRAY_TASK_ID]}
fi

export LCDB_INITIAL_CONFIGS=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/initial_configs.csv
export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/$LCDB_OPENML_ID
export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED

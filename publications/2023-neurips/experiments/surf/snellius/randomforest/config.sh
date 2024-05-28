#!/bin/bash

export LCDB_NUM_CONFIGS=10
export LCDB_WORKFLOW=lcdb.workflow.sklearn.RandomForestWorkflow
export LCDB_WORKFLOW_SEED=42
export LCDB_VALID_SEED=42
export LCDB_TEST_SEED=42
# Read the CSV file and load into an array
declare -a values
while IFS= read -r line || [[ -n "$line" ]]; do
    values+=("$line")
done < "./../datasets_to_test.csv"
export LCDB_OPENML_ID_ARRAY=(${values[@]})

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
  export LCDB_OPENML_ID=3
else
  export LCDB_OPENML_ID=${LCDB_OPENML_ID_ARRAY[SLURM_ARRAY_TASK_ID]}
fi

export LCDB_OUTPUT_WORKFLOW=$PWD/output/$LCDB_WORKFLOW
export LCDB_INITIAL_CONFIGS=$LCDB_OUTPUT_WORKFLOW/initial_configs.csv
export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW/$LCDB_OPENML_ID
export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED

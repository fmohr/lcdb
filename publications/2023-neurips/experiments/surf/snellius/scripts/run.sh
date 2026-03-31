#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --threads-per-core=1

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0

source ~/.bashrc
conda activate lcdb

# Disable GPUs explicitly
export CUDA_VISIBLE_DEVICES=""

# Re-source parse_core_assignments.sh to rebuild associative arrays in job context
# (associative arrays don't export through sbatch environment)
source "$path_to_snellius/scripts/parse_core_assignments.sh"

# Parse remaining OpenML IDs and get the one for this array task
IFS=',' read -r -a REMAINING_IDS <<< "$REMAINING_OPENML_IDS"
LCDB_OPENML_ID=${REMAINING_IDS[$SLURM_ARRAY_TASK_ID]}

# Look up configuration from associative arrays
NTOTRANKS=${OPENML_PARALLEL_TASKS[$LCDB_OPENML_ID]}
CPUS_PER_CONFIG=${OPENML_CORES_PER_TASK[$LCDB_OPENML_ID]}
LCDB_WORKFLOW_MEMORY_LIMIT_GB=${OPENML_MEMORY_PER_TASK[$LCDB_OPENML_ID]}

# Calculate memory limit in MB
LCDB_WORKFLOW_MEMORY_LIMIT_MB=$(echo "$LCDB_WORKFLOW_MEMORY_LIMIT_GB * 1024" | bc)
LCDB_WORKFLOW_MEMORY_LIMIT=$(printf "%.0f" "$LCDB_WORKFLOW_MEMORY_LIMIT_MB")

export NTOTRANKS
export CPUS_PER_CONFIG
export LCDB_WORKFLOW_MEMORY_LIMIT
export LCDB_WORKFLOW_MEMORY_LIMIT_GB

echo "OpenML ID: $LCDB_OPENML_ID"
echo "Parallel tasks: $NTOTRANKS"
echo "Cores per task: $CPUS_PER_CONFIG"
echo "Memory per task: $LCDB_WORKFLOW_MEMORY_LIMIT_GB GB"

#!!! CONFIGURATION - START
export timeout=-1
#!!! CONFIGURATION - END

IFS=' ' read -r -a VAL_SEEDS <<< "$VAL_SEEDS"
IFS=' ' read -r -a TEST_SEEDS <<< "$TEST_SEEDS"

echo "Validation seeds: ${VAL_SEEDS[*]}"
echo "Test seeds: ${TEST_SEEDS[*]}"

# Log directories (created by run_wrapper.sh)
export LOG_OUT_DIR="${output_path}/logs/${LCDB_WORKFLOW}/out"
export LOG_ERR_DIR="${output_path}/logs/${LCDB_WORKFLOW}/err"

export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW/$LCDB_OPENML_ID

for LCDB_VALID_SEED in "${VAL_SEEDS[@]}"; do
    for LCDB_TEST_SEED in "${TEST_SEEDS[@]}"; do
        export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED

        mkdir -p $LCDB_OUTPUT_RUN
        pushd $LCDB_OUTPUT_RUN

        # Creating the 'started' status file
        status="started"
        export STATUS_FILE_BASE="$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME/$LCDB_WORKFLOW-$LCDB_OPENML_ID-$LCDB_WORKFLOW_SEED-$LCDB_TEST_SEED-$LCDB_VALID_SEED"
        STATUS_FILE="$STATUS_FILE_BASE.$status"
        mkdir -p "$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME"

        if [ -f "$STATUS_FILE" ]; then
            echo "File $STATUS_FILE already exists. Skipping this configuration."
            gzip -f --best results.jsonl
            continue
        else
            echo "Creating status file: $STATUS_FILE"
            touch "$STATUS_FILE"

            # Run experiment with dynamic resource allocation
            # srun will allocate from the full node based on these parameters
            srun -n ${NTOTRANKS} \
                    --ntasks=${NTOTRANKS} \
                    --cpus-per-task=$CPUS_PER_CONFIG \
                    --threads-per-core=1 \
                    --exclusive \
                    --output=${LOG_OUT_DIR}/o:${LCDB_OPENML_ID}-w:${LCDB_WORKFLOW_SEED}-v:${LCDB_VALID_SEED}-t:${LCDB_TEST_SEED}.log \
                    --error=${LOG_ERR_DIR}/o:${LCDB_OPENML_ID}-w:${LCDB_WORKFLOW_SEED}-v:${LCDB_VALID_SEED}-t:${LCDB_TEST_SEED}.err \
                lcdb run \
                    --campaign $CAMPAIGN_NAME \
                    --openml-id $LCDB_OPENML_ID \
                    --workflow-class $LCDB_WORKFLOW \
                    --monotonic \
                    --max-evals $LCDB_NUM_CONFIGS \
                    --timeout $timeout \
                    --initial-configs $LCDB_INITIAL_CONFIGS \
                    --timeout-on-fit 300 \
                    --timeout-on-predict 60 \
                    --timeout-on-metrics 60 \
                    --workflow-seed $LCDB_WORKFLOW_SEED \
                    --workflow-memory-limit $LCDB_WORKFLOW_MEMORY_LIMIT \
                    --memory-patience 10 \
                    --ncpus $CPUS_PER_CONFIG \
                    --valid-seed $LCDB_VALID_SEED \
                    --no-exception-on-unsuitable-preprocessor \
                    --test-seed $LCDB_TEST_SEED \
                    --log-level info \
                    --evaluator mpicomm \
                    --epoch-schedule=power-2-0.25-0 

            # convert the csv format of deephyper into a jsonl file
            python $path_to_snellius/scripts/deephyper_csv_to_jsonl.py results.csv results.jsonl

            # gzip the json results
            gzip -f --best results.jsonl
        fi
    done 
done
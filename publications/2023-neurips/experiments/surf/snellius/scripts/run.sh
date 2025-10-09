#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --threads-per-core=1

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0 

source ~/.bashrc
conda activate lcdb

# Disable GPUs explicitly
export CUDA_VISIBLE_DEVICES=""

# === BIN_TAG awareness for per-bin internal logs ===
# Wrapper may export BIN_TAG (e.g., "1p6"); if missing, synthesize from memory limit.
if [[ -z "${BIN_TAG:-}" ]]; then
    if [[ -n "${LCDB_WORKFLOW_MEMORY_LIMIT_GB:-}" ]]; then
        export BIN_TAG="$(echo "$LCDB_WORKFLOW_MEMORY_LIMIT_GB" | tr '.' 'p')"
    else
        export BIN_TAG="unknown"
    fi
fi
# Logging directories
export LCDB_BIN_LOG_OUT="${output_path}/logs/${WORKFLOW_NAME}/bin_${BIN_TAG}/out"
export LCDB_BIN_LOG_ERR="${output_path}/logs/${WORKFLOW_NAME}/bin_${BIN_TAG}/err"
mkdir -p "$LCDB_BIN_LOG_OUT" "$LCDB_BIN_LOG_ERR"
# === END BIN_TAG ===

#!!! CONFIGURATION - START
export timeout=3500
export NTOTRANKS=$DESIRED_CORES
#!!! CONFIGURATION - END

echo "DEBUG: LCDB_OPENML_ARRAY_STRING='$LCDB_OPENML_ARRAY_STRING'"

IFS=' ' read -r -a VAL_SEEDS <<< "$VAL_SEEDS"
IFS=' ' read -r -a TEST_SEEDS <<< "$TEST_SEEDS"
IFS=',' read -r -a LCDB_OPENML_ID_ARRAY <<< "$LCDB_OPENML_ARRAY_STRING"
LCDB_OPENML_ID=${LCDB_OPENML_ID_ARRAY[$SLURM_ARRAY_TASK_ID]}
echo "Running experiment for OpenML ID: $LCDB_OPENML_ID"
echo "Validation seeds: ${VAL_SEEDS}"
echo "Test seeds: ${TEST_SEEDS}"

export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/$LCDB_OPENML_ID

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
            continue
        else
            echo "Creating status file: $STATUS_FILE"
            touch "$STATUS_FILE"

            # Run experiment
            srun -n ${NTOTRANKS} -N ${SLURM_JOB_NUM_NODES:-1} \
                    --cpus-per-task $CPUS_PER_CONFIG \
                    --threads-per-core 1 \
                    --exclusive \
                    --output=${LCDB_BIN_LOG_OUT}/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.log \
                    --error=${LCDB_BIN_LOG_ERR}/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.err \
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
                    --ncpus $CPUS_PER_CONFIG \
                    --valid-seed $LCDB_VALID_SEED \
                    --no-exception-on-unsuitable-preprocessor \
                    --test-seed $LCDB_TEST_SEED \
                    --log-level info \
                    --evaluator mpicomm \
                    --epoch-schedule=power-2-0.25-0 

            # convert the csv format of deephyper into a jsonl file
            python deephyper_csv_to_jsonl.py results.csv results.jsonl

            # gzip the json results
            gzip --best results.jsonl
        fi
    done 
done

#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --threads-per-core=1
# ^^^ Do NOT set --ntasks or --gpus here; your run_wrapper controls that.

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0

source ~/.bashrc
conda activate lcdbgpu

# ========================= CONFIGURATION =========================
export timeout=-1
export NTOTRANKS=${SLURM_NTASKS:-1}
export CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}

echo "Job launched with ${NTOTRANKS} rank(s) and ${CPUS_PER_TASK} CPU(s) per rank"
echo "GPUs allocated: ${SLURM_GPUS:-0}"
# ================================================================

# # Disable GPUs explicitly if not on GPU partition
# if [[ "$PARTITION_RUN" != "gpu_mig" && "$PARTITION_RUN" != "gpu_a100" ]]; then
#     export CUDA_VISIBLE_DEVICES=""
#     echo "Non-GPU partition detected; CUDA disabled."
# fi

# log directories created if not already present
mkdir -p "${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out"
mkdir -p "${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err"

IFS=' ' read -r -a VAL_SEEDS <<< "$VAL_SEEDS"
IFS=' ' read -r -a TEST_SEEDS <<< "$TEST_SEEDS"
IFS=',' read -r -a LCDB_OPENML_ID_ARRAY <<< "$LCDB_OPENML_ARRAY_STRING"
LCDB_OPENML_ID=${LCDB_OPENML_ID_ARRAY[$SLURM_ARRAY_TASK_ID]}

echo "Running experiment for OpenML ID: $LCDB_OPENML_ID"
echo "Validation seeds: ${VAL_SEEDS[*]}"
echo "Test seeds: ${TEST_SEEDS[*]}"

export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/$LCDB_OPENML_ID

for LCDB_VALID_SEED in "${VAL_SEEDS[@]}"; do
  for LCDB_TEST_SEED in "${TEST_SEEDS[@]}"; do
    export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED
    mkdir -p "$LCDB_OUTPUT_RUN"
    pushd "$LCDB_OUTPUT_RUN" >/dev/null

    status="started"
    export STATUS_FILE_BASE="$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME/$LCDB_WORKFLOW-$LCDB_OPENML_ID-$LCDB_WORKFLOW_SEED-$LCDB_TEST_SEED-$LCDB_VALID_SEED"
    STATUS_FILE="$STATUS_FILE_BASE.$status"
    mkdir -p "$(dirname "$STATUS_FILE")"

    if [ -f "$STATUS_FILE" ]; then
        echo "File $STATUS_FILE already exists. Skipping."
        popd >/dev/null
        continue
    fi

    echo "Creating status file: $STATUS_FILE"
    touch "$STATUS_FILE"
    

    # print setup of srun 
    echo "SETUP: srun -n ${NTOTRANKS} -N ${SLURM_JOB_NUM_NODES:-1} \
         --cpus-per-task=${CPUS_PER_TASK} \
         --gpus-per-task=1 \
         --threads-per-core=1 \
         --exclusive \
         --output=${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.log \
         --error=${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.err \
    "   

    # ---------------- GPU job execution ----------------
    srun -n ${NTOTRANKS} -N ${SLURM_JOB_NUM_NODES:-1} \
         --cpus-per-task=${CPUS_PER_TASK} \
         --gpus-per-task=1 \
         --threads-per-core=1 \
         --exclusive \
         --output=${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.log \
         --error=${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.err \
    lcdb run \
        --campaign $CAMPAIGN_NAME \
        --openml-id $LCDB_OPENML_ID \
        --workflow-class $LCDB_WORKFLOW \
        --monotonic \
        --max-evals $LCDB_NUM_CONFIGS \
        --timeout $timeout \
        --initial-configs $LCDB_INITIAL_CONFIGS \
        --timeout-on-fit 300 \
        --workflow-seed $LCDB_WORKFLOW_SEED \
        --workflow-memory-limit $LCDB_WORKFLOW_MEMORY_LIMIT \
        --valid-seed $LCDB_VALID_SEED \
        --test-seed $LCDB_TEST_SEED \
        --no-exception-on-unsuitable-preprocessor \
        --log-level debug \
        --evaluator process \
        --num-epochs=5 \
        --epoch-schedule=power-2-0.25-0
    # ---------------------------------------------------

    gzip --best results.csv 2>/dev/null || echo "No results.csv found to gzip."
    popd >/dev/null
  done
done

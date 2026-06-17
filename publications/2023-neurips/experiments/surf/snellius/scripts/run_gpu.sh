#!/bin/bash
#SBATCH --threads-per-core=1
# ^^^ Do NOT set --ntasks or --gpus here; your run_wrapper controls that.

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0

source ~/.bashrc
conda activate lcdbgpu

# ========================= CONFIGURATION =========================
export timeout=-1

# Re-source parse_core_assignments.sh to rebuild associative arrays in job context
# (associative arrays don't export through sbatch environment)
source "$path_to_snellius/scripts/parse_core_assignments.sh"

# Parse remaining OpenML IDs and get the one for this array task
IFS=',' read -r -a REMAINING_IDS <<< "$REMAINING_OPENML_IDS"
LCDB_OPENML_ID=${REMAINING_IDS[$SLURM_ARRAY_TASK_ID]}

# Look up configuration from associative arrays
export NTOTRANKS=${OPENML_PARALLEL_TASKS[$LCDB_OPENML_ID]}
export CPUS_PER_TASK=${OPENML_CORES_PER_TASK[$LCDB_OPENML_ID]}
LCDB_WORKFLOW_MEMORY_LIMIT_GB=${OPENML_MEMORY_PER_TASK[$LCDB_OPENML_ID]}

# Calculate memory limit in MB
LCDB_WORKFLOW_MEMORY_LIMIT_MB=$(echo "$LCDB_WORKFLOW_MEMORY_LIMIT_GB * 1024" | bc)
LCDB_WORKFLOW_MEMORY_LIMIT=$(printf "%.0f" "$LCDB_WORKFLOW_MEMORY_LIMIT_MB")

export LCDB_WORKFLOW_MEMORY_LIMIT
export LCDB_WORKFLOW_MEMORY_LIMIT_GB

# Calculate number of GPUs needed: NTOTRANKS - 1 (master rank doesn't compute)
NUM_GPUS=$((NTOTRANKS - 1))
export NUM_GPUS

if (( NUM_GPUS < 1 )); then
  echo "GPU runs need at least 2 ranks: one CPU master and at least one GPU worker. Got NTOTRANKS=$NTOTRANKS" >&2
  exit 1
fi

echo "OpenML ID: $LCDB_OPENML_ID"
echo "Job launched with ${NTOTRANKS} rank(s) and ${CPUS_PER_TASK} CPU(s) per rank"
echo "GPUs needed: ${NUM_GPUS} (${NTOTRANKS} ranks - 1 master)"
echo "Memory per task: $LCDB_WORKFLOW_MEMORY_LIMIT_GB GB"
# ================================================================

# Log directories (created by run_wrapper.sh)
export LOG_OUT_DIR="${output_path}/logs/${LCDB_WORKFLOW}/out"
export LOG_ERR_DIR="${output_path}/logs/${LCDB_WORKFLOW}/err"

IFS=' ' read -r -a VAL_SEEDS <<< "$VAL_SEEDS"
IFS=' ' read -r -a TEST_SEEDS <<< "$TEST_SEEDS"

echo "Validation seeds: ${VAL_SEEDS[*]}"
echo "Test seeds: ${TEST_SEEDS[*]}"

export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW/$LCDB_OPENML_ID

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

    lcdb_run_command=(
      lcdb run
      --campaign "$CAMPAIGN_NAME"
      --openml-id "$LCDB_OPENML_ID"
      --workflow-class "$LCDB_WORKFLOW"
      --monotonic
      --max-evals "$LCDB_NUM_CONFIGS"
      --timeout "$timeout"
      --initial-configs "$LCDB_INITIAL_CONFIGS"
      --timeout-on-fit 900
      --workflow-seed "$LCDB_WORKFLOW_SEED"
      --workflow-memory-limit "$LCDB_WORKFLOW_MEMORY_LIMIT"
      --memory-patience 10
      --ncpus "$CPUS_PER_TASK"
      --valid-seed "$LCDB_VALID_SEED"
      --test-seed "$LCDB_TEST_SEED"
      --no-exception-on-unsuitable-preprocessor
      --log-level info
      --evaluator mpicomm
      --epoch-schedule=power-2-0.25-0
    )

    # ---------------- GPU job execution ----------------
    # Rank 0 is the CPU-only Deephyper master; each worker rank sees one GPU.
    # The batch allocation reserves CPUs for GPU workers; the lightweight master shares that allocation.
    srun -n ${NTOTRANKS} \
         --ntasks=${NTOTRANKS} \
         --cpus-per-task=${CPUS_PER_TASK} \
         --gpus=${NUM_GPUS} \
         --threads-per-core=1 \
       --overcommit \
         --output=${LOG_OUT_DIR}/o:${LCDB_OPENML_ID}-w:${LCDB_WORKFLOW_SEED}-v:${LCDB_VALID_SEED}-t:${LCDB_TEST_SEED}.log \
         --error=${LOG_ERR_DIR}/o:${LCDB_OPENML_ID}-w:${LCDB_WORKFLOW_SEED}-v:${LCDB_VALID_SEED}-t:${LCDB_TEST_SEED}.err \
         bash -c '
          local_rank="${SLURM_LOCALID:-${SLURM_PROCID:-0}}"
          inherited_cuda="${CUDA_VISIBLE_DEVICES:-}"

          if (( local_rank == 0 )); then
            export CUDA_VISIBLE_DEVICES=""
          else
            worker_index=$((local_rank - 1))
            IFS=, read -r -a allocated_gpus <<< "$inherited_cuda"
            export CUDA_VISIBLE_DEVICES="${allocated_gpus[$worker_index]:-$worker_index}"
          fi

          exec "$@"
         ' bash "${lcdb_run_command[@]}"
    srun_status=$?
    if (( srun_status != 0 )); then
      echo "lcdb run failed for OpenML ID ${LCDB_OPENML_ID} with exit code ${srun_status}" >&2
      exit "$srun_status"
    fi
    # ---------------------------------------------------

    # gzip the json results if the status file completed exists
    if [ -f "$STATUS_FILE_BASE.completed" ]; then
        gzip -f --best results.jsonl
    fi

    popd >/dev/null
  done
done

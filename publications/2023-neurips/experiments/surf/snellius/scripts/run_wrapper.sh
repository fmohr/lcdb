#!/bin/bash
# set -xe
source ~/.bashrc
conda activate lcdb

export SIMPLE_GPU_MODE=true  # Set to false to use normal gpu_mig mode

export PARTITION_CREATE="rome"
export PARTITION_RUN="genoa"  # or "genoa" for CPU jobs, "gpu_a100" for GPU jobs

# if first argument is densenn then partition run is gpu_a100
if [ "$1" == "densenn" ]; then
    export PARTITION_RUN="gpu_mig"
fi

export path_to_snellius=$(pwd)
export output_path="/gpfs/nvme1/0/prjs1064/LCDB2"

# Workflow name from CLI arg
REQ_WORKFLOW_NAME="${1:-knn}"
export WORKFLOW_NAME="$REQ_WORKFLOW_NAME"

echo "PARTITION_RUN: $PARTITION_RUN"
echo "WORKFLOW_NAME: $WORKFLOW_NAME"
echo ""

# Source config to get workflow mapping and defaults
source "$path_to_snellius/scripts/config.sh"

# Parse core assignments and bin datasets by resource requirements
source "$path_to_snellius/scripts/parse_core_assignments.sh"

# Export for create.sh
export LCDB_OPENML_ID_ARRAY=("${ALL_OPENML_IDS[@]}")

# Create log directories using full workflow class name
log_out_dir="${output_path}/logs/${LCDB_WORKFLOW}/out"
log_err_dir="${output_path}/logs/${LCDB_WORKFLOW}/err"
mkdir -p "$log_out_dir" "$log_err_dir"

# checkpoints/status
checkpoint_dir="$LCDB_OUTPUT_WORKFLOW/exp-checkpoints/$CAMPAIGN_NAME"
mkdir -p "$checkpoint_dir"
export CAMPAIGN_STATUS_FILE="$checkpoint_dir/$LCDB_WORKFLOW.general"

# Submit create job
workflow_create_job_id=$(sbatch --export=ALL \
    --output="${log_out_dir}/create_%j.log" \
    --error="${log_err_dir}/create_%j.err" \
    --chdir="${output_path}" \
    --parsable \
    --partition="$PARTITION_CREATE" \
    --job-name="${CAMPAIGN_NAME}-create-wf_${WORKFLOW_NAME}" \
    scripts/create.sh)

echo "Submitted create.sh for workflow '${WORKFLOW_NAME}' with Job ID: $workflow_create_job_id"
echo ""

# Filter already-submitted datasets
declare -a remaining_ids=()
for openml_id in "${ALL_OPENML_IDS[@]}"; do
    status="submitted"
    STATUS_FILE="$checkpoint_dir/openmlID-$openml_id.$LCDB_WORKFLOW.$status"
    if (set -o noclobber; : > "$STATUS_FILE") 2> /dev/null; then
        remaining_ids+=("$openml_id")
    else
        echo "Skipping OpenML ID $openml_id (already submitted)"
    fi
done

if [ ${#remaining_ids[@]} -eq 0 ]; then
    echo "No new datasets to process."
    exit 0
fi

echo "Submitting ${#remaining_ids[@]} datasets out of ${#ALL_OPENML_IDS[@]} total"

# Export remaining IDs as comma-separated string for the job array
# The associative arrays (OPENML_PARALLEL_TASKS, etc.) are already exported from parse_core_assignments.sh
IFS=',' eval 'export REMAINING_OPENML_IDS="${remaining_ids[*]}"'

# ---------- RUN ----------
script="$path_to_snellius/scripts/run.sh"
jobname="${CAMPAIGN_NAME}-run-${WORKFLOW_NAME}"

if [ "$PARTITION_RUN" == "gpu_mig" ] || [ "$PARTITION_RUN" == "gpu_a100" ]; then
    if [ "$SIMPLE_GPU_MODE" == "true" ]; then
        script="$path_to_snellius/scripts/run_gpu.sh"

        # Calculate max resources needed across all remaining datasets
        # (job arrays require uniform resources across all array tasks)
        max_parallel_tasks=0
        max_cpus_per_task=0
        for id in "${remaining_ids[@]}"; do
            parallel=${OPENML_PARALLEL_TASKS[$id]}
            cpus=${OPENML_CORES_PER_TASK[$id]}
            if (( parallel > max_parallel_tasks )); then
                max_parallel_tasks=$parallel
            fi
            if (( cpus > max_cpus_per_task )); then
                max_cpus_per_task=$cpus
            fi
        done

        # Calculate GPUs needed: max_parallel_tasks - 1 (master rank doesn't compute)
        # max_gpus=$((max_parallel_tasks - 1))
        # NUM_GPU_WORKERS=$max_gpus
        NUM_GPU_WORKERS=$((max_parallel_tasks - 1))


        echo "Max resources for GPU job array: ${max_parallel_tasks} ranks, ${max_cpus_per_task} CPUs/rank, ${max_gpus} GPUs"

            # --mem-per-cpu=4G \
            # --gpus=$max_gpus \


            # --gpus-per-task=1 \
            # --ntasks=$max_parallel_tasks \
            # --cpus-per-task=$max_cpus_per_task \
        run_job_id=$(sbatch \
            --partition="$PARTITION_RUN" \
            --dependency=afterok:$workflow_create_job_id \
            --time=6:00:00 \
            --nodes=1 \
            --ntasks=$NUM_GPU_WORKERS \
            --gpus=$NUM_GPU_WORKERS \
            --cpus-per-task=$max_cpus_per_task \
            --job-name="${jobname}" \
            --array=0-$((${#remaining_ids[@]} - 1)) \
            --export=ALL \
            --chdir="$output_path" \
            --output="${log_out_dir}/job_%A_%a.log" \
            --error="${log_err_dir}/job_%A_%a.err" \
            --parsable \
            "$script")
        echo "Submitted GPU job array with ID: $run_job_id"
    else
        echo "Advanced MIG mode not enabled in this template."
        exit 1
    fi
else
    export CUDA_VISIBLE_DEVICES=""

    run_job_id=$(sbatch --export=ALL \
        --job-name="$jobname" \
        --dependency=afterok:$workflow_create_job_id \
        --exclusive \
        --nodes=1 \
        --mem=0 \
        --array=0-$((${#remaining_ids[@]} - 1)) \
        --partition=$PARTITION_RUN \
        --chdir=$output_path \
        --output="${log_out_dir}/job_%A_%a.log" \
        --error="${log_err_dir}/job_%A_%a.err" \
        --parsable \
        "$script")
    echo "Submitted CPU job array with I·D: $run_job_id"
fi

# Collect result files for campaign upload
declare -a ALL_RESULT_FILES=()
for id in "${remaining_ids[@]}"; do
    IFS=' ' read -r -a VAL_SEEDS_ARRAY <<< "$VAL_SEEDS"
    IFS=' ' read -r -a TEST_SEEDS_ARRAY <<< "$TEST_SEEDS"
    for val_seed in "${VAL_SEEDS_ARRAY[@]}"; do
        for test_seed in "${TEST_SEEDS_ARRAY[@]}"; do
            result_path="${LCDB_OUTPUT_WORKFLOW}/$id/${val_seed}-${test_seed}-${LCDB_WORKFLOW_SEED}/results.jsonl.gz"
            ALL_RESULT_FILES+=("$result_path")
        done
    done
done

echo ""

# -------------------------------------------------------------------------
# Submit campaign upload job with dependency on run job
# -------------------------------------------------------------------------
echo ""
echo "==== Submitting campaign upload job ===="
echo "Collected ${#ALL_RESULT_FILES[@]} result files to upload."
echo "Waiting for run job $run_job_id to complete."

# Set up environment path for pCloud authentication
export ENV_PATH="$path_to_snellius/../../.env"
echo "Using .env path: $ENV_PATH"

# Create campaign log directories using full workflow class name
campaign_out_dir="${output_path}/logs-campaign/${LCDB_WORKFLOW}/out"
campaign_err_dir="${output_path}/logs-campaign/${LCDB_WORKFLOW}/err"
mkdir -p "$campaign_out_dir" "$campaign_err_dir"

# Submit campaign job with dependency on run job
    # --mail-type=END,FAIL \
    # --mail-user="felixmoh@unisabana.edu.co" \
campaign_job_id=$(sbatch \
    --export=ALL,LCDB_WORKFLOW="$LCDB_WORKFLOW",CAMPAIGN_NAME="$CAMPAIGN_NAME",ENV_PATH="$ENV_PATH" \
    --job-name="campaign_${WORKFLOW_NAME}" \
    --dependency="afterany:$run_job_id" \
    --output="${campaign_out_dir}/campaign.log" \
    --error="${campaign_err_dir}/campaign.err" \
    --chdir="${output_path}" \
    --parsable \
    scripts/campaign.sh "${ALL_RESULT_FILES[@]}")

echo "Submitted campaign upload job with ID: $campaign_job_id"
echo "Campaign will start after all run jobs complete."

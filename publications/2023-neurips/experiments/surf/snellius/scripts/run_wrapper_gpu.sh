#!/bin/bash
# set -xe
source ~/.bashrc
conda activate lcdbgpu

export SIMPLE_GPU_MODE=true  # Set to false to use normal gpu_mig mode

export PARTITION_CREATE="rome"
export PARTITION_RUN="gpu_mig"  # or "genoa" if you want CPU jobs "gpu_mig"

export path_to_snellius=$(pwd)
export output_path="/gpfs/nvme1/0/prjs1064/LCDB2"

# Optional: workflow via CLI arg
REQ_WORKFLOW_NAME="${1:-}"

# helper to find per-workflow bins csv
find_bins_csv() {
    local wname="$1"
    shopt -s nullglob
    local matches=( "$path_to_snellius"/*"${wname}"*bins*.csv )
    shopt -u nullglob
    if (( ${#matches[@]} == 0 )); then
        echo ""
    else
        echo "${matches[0]}"
    fi
}

# Load config for mapping + defaults
if [[ -n "$REQ_WORKFLOW_NAME" ]]; then
    export WORKFLOW_NAME="$REQ_WORKFLOW_NAME"
fi
source "$path_to_snellius/scripts/config.sh"

if [[ -z "$REQ_WORKFLOW_NAME" ]]; then
    REQ_WORKFLOW_NAME="$WORKFLOW_NAME"
fi

# per-workflow CSV (id,bin)
if [[ -z "${WORKFLOW_DATASET_CSV:-}" ]]; then
    WORKFLOW_DATASET_CSV="$(find_bins_csv "$REQ_WORKFLOW_NAME")"
fi
if [[ -z "$WORKFLOW_DATASET_CSV" || ! -f "$WORKFLOW_DATASET_CSV" ]]; then
    echo "No *bins*.csv found for workflow '$REQ_WORKFLOW_NAME' (looked at '$WORKFLOW_DATASET_CSV')."
    exit 1
fi
echo "Using per-bin CSV: $WORKFLOW_DATASET_CSV"

# group ids by bin
declare -A BIN_TO_IDS
declare -a UNIQUE_BINS

# parse the CSV (id,mem)
while IFS=, read -r raw_id raw_mem _ || [[ -n "$raw_id" ]]; do
    # Remove any carriage returns and trim whitespace
    id="$(echo "$raw_id" | tr -d '\r' | xargs)"
    mem="$(echo "$raw_mem" | tr -d '\r' | xargs)"

    # Debug to confirm parsing
    echo ">> DEBUG: parsed id='$id' mem='$mem'"

    [[ -z "$id" || -z "$mem" ]] && continue

    # id must be digits
    if ! [[ "$id" =~ ^[0-9]+$ ]]; then
        echo ">> DEBUG: skipping line (id not numeric): $raw_id,$raw_mem"
        continue
    fi
    # mem must be number or float
    if ! [[ "$mem" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo ">> DEBUG: skipping line (mem not numeric): $raw_id,$raw_mem"
        continue
    fi

    if [[ -z "${BIN_TO_IDS[$mem]:-}" ]]; then
        UNIQUE_BINS+=("$mem")
        BIN_TO_IDS["$mem"]="$id"
    else
        BIN_TO_IDS["$mem"]="${BIN_TO_IDS[$mem]},$id"
    fi
done < <(cat "$WORKFLOW_DATASET_CSV" | LC_ALL=C tr -d '\r')


if (( ${#UNIQUE_BINS[@]} == 0 )); then
    echo "No (OpenML ID, memory bin) rows parsed from $WORKFLOW_DATASET_CSV"
    exit 1
fi

for BIN in "${UNIQUE_BINS[@]}"; do
    export DESIRED_MEMORY_GB="$BIN"         # <-- export so config.sh sees it
    export BIN_TAG="${BIN//./p}"

    IFS=',' read -r -a LCDB_OPENML_ID_ARRAY <<< "${BIN_TO_IDS[$BIN]}"
    export LCDB_OPENML_ID_ARRAY

    unset WORKFLOW_DATASET_CSV
    source "$path_to_snellius/scripts/config.sh"



    # per-bin log dirs
    BIN_LOG_DIR="${output_path}/logs_test/${WORKFLOW_NAME}/bin_${BIN_TAG}"
    mkdir -p "${BIN_LOG_DIR}/out" "${BIN_LOG_DIR}/err"

    # checkpoints/status
    LCDB_WORKFOW_DIR=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB
    checkpoint_dir="$LCDB_WORKFOW_DIR/exp-checkpoints/$CAMPAIGN_NAME"
    mkdir -p "$checkpoint_dir"
    export CAMPAIGN_STATUS_FILE="$checkpoint_dir/$LCDB_WORKFLOW.general"

    # de-dup submissions
    remaining_ids=()
    for openml_id in "${LCDB_OPENML_ID_ARRAY[@]}"; do
        export LCDB_OPENML_ID=$openml_id
        status="submitted"
        export STATUS_FILE_BASE="$checkpoint_dir/openmlID-$LCDB_OPENML_ID.$LCDB_WORKFLOW"
        STATUS_FILE="$STATUS_FILE_BASE.$status"
        if (set -o noclobber; : > "$STATUS_FILE") 2> /dev/null; then
            echo "Created status file: $STATUS_FILE"
            remaining_ids+=("$openml_id")
        else
            echo "File $STATUS_FILE already exists. Skipping this configuration."
        fi
    done
    export LCDB_OPENML_ID_ARRAY=("${remaining_ids[@]}")
    if [ ${#LCDB_OPENML_ID_ARRAY[@]} -eq 0 ]; then
        echo "No OpenML IDs to process for bin $BIN. Skipping."
        continue
    fi

    export LCDB_OPENML_ARRAY_STRING
    LCDB_OPENML_ARRAY_STRING=$(IFS=,; echo "${LCDB_OPENML_ID_ARRAY[*]}")

    # ---------- CREATE ----------
    create_job_id=$(sbatch --export=ALL \
        --output="${BIN_LOG_DIR}/out/create_%j.log" \
        --error="${BIN_LOG_DIR}/err/create_%j.err" \
        --chdir="${output_path}" \
        --parsable \
        --partition="$PARTITION_CREATE" \
        --job-name="${CAMPAIGN_NAME}-create-wf_${WORKFLOW_NAME}-bin_${BIN_TAG}" \
        scripts/create.sh)
    echo "Submitted create.sh (bin ${BIN} GB) with Job ID: $create_job_id"

    # ---------- RUN ----------
    script="$path_to_snellius/scripts/run_gpu.sh"
    jobname="${CAMPAIGN_NAME}-run-wf_${WORKFLOW_NAME}-bin_${BIN_TAG}"

    echo $LCDB_WORKFLOW_MEMORY_LIMIT

    if [ "$PARTITION_RUN" == "gpu_mig" ]; then
        if [ "$SIMPLE_GPU_MODE" == "true" ]; then
            echo "Submitting simplified GPU job for bin $BIN..."
            CPUS_PER_TASK=9
            GPUS_PER_TASK=1
            TOTAL_IDS=${#LCDB_OPENML_ID_ARRAY[@]}
            sbatch --partition=gpu_mig \
                --export=ALL \
                --ntasks=1 \
                --cpus-per-task=$CPUS_PER_TASK \
                --gpus-per-task=$GPUS_PER_TASK \
                # --mem-per-gpu=60G \
                --dependency=afterok:$create_job_id \
                --job-name="${jobname}_simple" \
                --array=0-$(($TOTAL_IDS - 1)) \
                --chdir=$output_path \
                --output="${BIN_LOG_DIR}/out/runGPU_simple_%A_%a.log" \
                --error="${BIN_LOG_DIR}/err/runGPU_simple_%A_%a.err" \
                "$script"
        else
            echo "Advanced MIG mode not enabled in this template."
            exit 1
        fi
    else
        echo "Submitting CPU job for bin $BIN..."
        export CUDA_VISIBLE_DEVICES=""
        sbatch --export=ALL --job-name="$jobname" \
            --dependency=afterok:$create_job_id \
            --exclusive \
            --ntasks=$DESIRED_CORES \
            --array=0-$((${#LCDB_OPENML_ID_ARRAY[@]} - 1)) \
            --mem=0 \
            --nodes=$NODES \
            --partition=$PARTITION_RUN \
            --chdir=$output_path \
            --output="${BIN_LOG_DIR}/out/run_%A_%a.log" \
            --error="${BIN_LOG_DIR}/err/run_%A_%a.err" \
            "$script"
    fi
done

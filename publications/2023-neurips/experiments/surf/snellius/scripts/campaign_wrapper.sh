#!/bin/bash
# Wrapper for uploading results to pCloud for a given workflow

source ~/.bashrc
conda activate lcdb

export path_to_snellius=$(pwd)
export output_path="/gpfs/nvme1/0/prjs1064/LCDB2"

# -------------------------------------------------------------------------
# Workflow → class mapping
# -------------------------------------------------------------------------
declare -A mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
    ["densenn"]="lcdb.workflow.keras.DenseNNWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
)

source "$path_to_snellius/scripts/config.sh"

log_dir="$output_path/logs/$WORKFLOW_NAME"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper_campaign.log") 2>&1

echo "==== Starting campaign wrapper for '$WORKFLOW_NAME' ===="
echo "Campaign name: $CAMPAIGN_NAME"

# -------------------------------------------------------------------------
# Determine list of datasets and bins
# -------------------------------------------------------------------------
bins_csv="$path_to_snellius/bins_config/${WORKFLOW_NAME}_bins.csv"
if [[ ! -f "$bins_csv" ]]; then
    echo "Error: expected $bins_csv not found"
    exit 1
fi

declare -A BIN_TO_IDS
declare -a UNIQUE_BINS

while IFS=, read -r raw_id raw_mem _ || [[ -n "$raw_id" ]]; do
    id="$(echo "$raw_id" | tr -d '\r' | xargs)"
    mem="$(echo "$raw_mem" | tr -d '\r' | xargs)"
    [[ -z "$id" || -z "$mem" ]] && continue
    if [[ -z "${BIN_TO_IDS[$mem]:-}" ]]; then
        UNIQUE_BINS+=("$mem")
        BIN_TO_IDS["$mem"]="$id"
    else
        BIN_TO_IDS["$mem"]="${BIN_TO_IDS[$mem]},$id"
    fi
done < "$bins_csv"

echo "Found ${#UNIQUE_BINS[@]} memory bins in $bins_csv"

# -------------------------------------------------------------------------
# Collect result files (jsonl.gz only, using rounded bin dirs)
# -------------------------------------------------------------------------
result_files=()
workflow_seed=$LCDB_WORKFLOW_SEED

round() {
    # round to nearest integer (like Python round)
    local num="$1"
    local rounded
    rounded=$(awk -v n="$num" 'BEGIN { printf "%d", (n>=0)?int(n+0.5):int(n-0.5) }')
    echo "$rounded"
}

for BIN in "${UNIQUE_BINS[@]}"; do
    rounded_bin=$(round "$BIN")
    IFS=',' read -r -a ids <<< "${BIN_TO_IDS[$BIN]}"

    for id in "${ids[@]}"; do
        for val_seed in "${VAL_SEEDS[@]}"; do
            for test_seed in "${TEST_SEEDS[@]}"; do
                result_path="${LCDB_OUTPUT_WORKFLOW}-${rounded_bin}/$id/${val_seed}-${test_seed}-${workflow_seed}/results.jsonl.gz"
                result_files+=("$result_path")
            done
        done
    done
done

echo "Collected ${#result_files[@]} result files to upload."

# -------------------------------------------------------------------------
# Prepare environment and submit upload job
# -------------------------------------------------------------------------
current_dir=$(pwd)
export ENV_PATH="$current_dir/../../../.env"
echo "Using .env path: $ENV_PATH"

out_dir="${output_path}/logs-campaign/${WORKFLOW_NAME}/out"
err_dir="${output_path}/logs-campaign/${WORKFLOW_NAME}/err"
mkdir -p "$out_dir" "$err_dir"

sbatch --export=ALL --job-name="campaigns_${WORKFLOW_NAME}" \
    --output="${out_dir}/campaign.log" \
    --error="${err_dir}/campaign.err" \
    --chdir="${output_path}" \
    scripts/campaign.sh "${result_files[@]}"

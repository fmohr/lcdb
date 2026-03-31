#!/bin/bash
# Wrapper for uploading existing results to pCloud for a given workflow

source ~/.bashrc
conda activate lcdb

export path_to_snellius=$(pwd)
export output_path="/gpfs/nvme1/0/prjs1064/LCDB2"

# Workflow name from CLI arg
# REQ_WORKFLOW_NAME="${1:-knn}"
# export WORKFLOW_NAME="$REQ_WORKFLOW_NAME"

# echo "WORKFLOW_NAME: $WORKFLOW_NAME"
echo ""

# Source config to resolve LCDB_WORKFLOW, LCDB_OUTPUT_WORKFLOW, CAMPAIGN_NAME, seeds, etc.
source "$path_to_snellius/scripts/config.sh"

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Create log directories using full workflow class name
log_dir="$output_path/logs/$LCDB_WORKFLOW"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper_campaign.log") 2>&1

# echo "==== Starting campaign wrapper for '$WORKFLOW_NAME' ===="
echo "Resolved LCDB_WORKFLOW: $LCDB_WORKFLOW"
echo "Campaign name: $CAMPAIGN_NAME"
echo ""

# Reuse the same dataset parsing logic as run_wrapper.sh
source "$path_to_snellius/scripts/parse_core_assignments.sh"

if [ ${#ALL_OPENML_IDS[@]} -eq 0 ]; then
    echo "No datasets found in ALL_OPENML_IDS."
    exit 1
fi

echo "Found ${#ALL_OPENML_IDS[@]} datasets"

# Collect result files exactly like run_wrapper.sh
declare -a ALL_RESULT_FILES=()

IFS=' ' read -r -a VAL_SEEDS_ARRAY <<< "$VAL_SEEDS"
IFS=' ' read -r -a TEST_SEEDS_ARRAY <<< "$TEST_SEEDS"

for id in "${ALL_OPENML_IDS[@]}"; do
    for val_seed in "${VAL_SEEDS_ARRAY[@]}"; do
        for test_seed in "${TEST_SEEDS_ARRAY[@]}"; do
            result_path="${LCDB_OUTPUT_WORKFLOW}/$id/${val_seed}-${test_seed}-${LCDB_WORKFLOW_SEED}/results.jsonl.gz"
            ALL_RESULT_FILES+=("$result_path")
        done
    done
done

echo "Collected ${#ALL_RESULT_FILES[@]} result files to upload."

# Set up environment path for pCloud authentication
export ENV_PATH="$path_to_snellius/../../.env"
echo "Using .env path: $ENV_PATH"

# Create campaign log directories using full workflow class name
campaign_out_dir="${output_path}/logs-campaign/${LCDB_WORKFLOW}/out"
campaign_err_dir="${output_path}/logs-campaign/${LCDB_WORKFLOW}/err"
mkdir -p "$campaign_out_dir" "$campaign_err_dir"

campaign_job_id=$(sbatch \
    --export=ALL,LCDB_WORKFLOW="$LCDB_WORKFLOW",CAMPAIGN_NAME="$CAMPAIGN_NAME",ENV_PATH="$ENV_PATH" \
    --job-name="campaign_${LCDB_WORKFLOW}" \
    --output="${campaign_out_dir}/campaign.log" \
    --error="${campaign_err_dir}/campaign.err" \
    --chdir="${output_path}" \
    --parsable \
    scripts/campaign.sh "${ALL_RESULT_FILES[@]}")

echo "Submitted campaign upload job with ID: $campaign_job_id"
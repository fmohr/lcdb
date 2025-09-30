#!/bin/bash

source ~/.bashrc
conda activate lcdb

export path_to_snellius=$(pwd)
export output_path="/gpfs/nvme1/0/prjs1064/LCDB2"

# Workflow mapping
declare -A mapping
mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
    ["densenn"]="lcdb.workflow.keras.DenseNNWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
)

source "$path_to_snellius/scripts/config.sh"

log_dir="$output_path/logs/$WORKFLOW_NAME-$MEM"
echo "Log directory: $log_dir"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper_campaign.log") 2>&1


echo "Campaign name is '$CAMPAIGN_NAME'"


source "$path_to_snellius/scripts/config.sh"

workflow_seed=$LCDB_WORKFLOW_SEED


echo "Processing workflow '$LCDB_WORKFLOW' with output path '$LCDB_OUTPUT_WORKFLOW'-$LCDB_WORKFLOW_MEMORY_LIMIT_GB'"

result_files=()
for id in "${LCDB_OPENML_ID_ARRAY[@]}"; do

    # Collect all result files for the given dataset id
    for val_seed in "${VAL_SEEDS[@]}"; do
        for test_seed in "${TEST_SEEDS[@]}"; do
            result_files+=("$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/$id/$val_seed-$test_seed-$workflow_seed/results.csv.gz")
        done
    done
done

# Define the path to the .env file (e.g., one level up)
current_dir=$(pwd)
export ENV_PATH="$current_dir/../../../.env"  
echo "Env path $ENV_PATH"

# Submit the job with the list of result files for this dataset
sbatch --export=ALL --job-name="campaigns_${WORKFLOW_NAME}" \
    --output=${output_path}/logs-campaign/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/campaign.log \
    --error=${output_path}/logs-campaign/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/campaign.err \
    --chdir=${output_path} \
    scripts/campaign.sh "${result_files[@]}"

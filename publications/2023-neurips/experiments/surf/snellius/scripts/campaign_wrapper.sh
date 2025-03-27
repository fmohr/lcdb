#!/bin/bash

source ~/.bashrc
conda activate lcdb

# Workflow mappings
declare -A mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
)

# Load configuration
CONFIG_FILE="scripts/config.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read the workflow name from config.yaml and remove any extra quotes or spaces
WORKFLOW_NAME=$(yq -r '.workflow_name' "$CONFIG_FILE")

# Ensure that the workflow name exists in the mapping
if [[ -n "${mapping[$WORKFLOW_NAME]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$WORKFLOW_NAME]}
    export LCDB_OUTPUT_WORKFLOW=$PWD/$WORKFLOW_NAME/output/$LCDB_WORKFLOW
else
    echo "Invalid workflow name: '$WORKFLOW_NAME'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Read memory and core configuration
export MEM=$(yq -r '.desired_memory_GB' "$CONFIG_FILE")
val_seeds=($(yq -r '.val_seeds[]' "$CONFIG_FILE"))
test_seeds=($(yq -r '.test_seeds[]' "$CONFIG_FILE"))

log_dir="$PWD/logs"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper_campaign.log") 2>&1

export campaign_name="data_probing-$MEM"

echo "Campaign name is '$campaign_name'"


source scripts/config.sh

workflow_seed=$LCDB_WORKFLOW_SEED


echo "Processing workflow '$LCDB_WORKFLOW' with output path '$LCDB_OUTPUT_WORKFLOW'"

result_files=()
for id in "${LCDB_OPENML_ID_ARRAY[@]}"; do

    # Collect all result files for the given dataset id
    for val_seed in "${val_seeds[@]}"; do
        for test_seed in "${test_seeds[@]}"; do
            result_files+=("$LCDB_OUTPUT_WORKFLOW-$MEM/$id/$val_seed-$test_seed-$workflow_seed/results.csv.gz")
        done
    done
done

# Submit the job with the list of result files for this dataset
sbatch --export=ALL --job-name="campaigns_${WORKFLOW_NAME}" \
    --output=logs/%x/logs/out/%x_workflow-${WORKFLOW_NAME}.log \
    --error=logs/%x/logs/err/%x_workflow-${WORKFLOW_NAME}.err \
    scripts/campaign.sh "${result_files[@]}"

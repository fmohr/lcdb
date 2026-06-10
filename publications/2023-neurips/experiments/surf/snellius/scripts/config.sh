#!/bin/bash
source ~/.bashrc
conda activate lcdb

# *********Loading Configuration*********

# Cluster configuration
export CPUS_PER_NODE=192  # genoa nodes

# Default configuration values
DEFAULT_WORKFLOW_NAME="densenn"
DEFAULT_CAMPAIGN_NAME="tabarena-gpu"
DEFAULT_CONFIG_NUM=50
DEFAULT_WORKFLOW_SEED=42
DEFAULT_VAL_SEEDS="0"
DEFAULT_TEST_SEEDS="0"

# workflow mapping
declare -A mapping
mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
    ["densenn"]="lcdb.workflow.keras.DenseNNWorkflow"
)

# If WORKFLOW_NAME not already set by wrapper, use default
if [[ -z "$WORKFLOW_NAME" ]]; then
    export WORKFLOW_NAME="$DEFAULT_WORKFLOW_NAME"
fi

# If CAMPAIGN_NAME not already set by wrapper, use default
if [[ -z "$CAMPAIGN_NAME" ]]; then
    export CAMPAIGN_NAME="$DEFAULT_CAMPAIGN_NAME"
fi

# check that the workflow name exists
if [[ -n "${mapping[$WORKFLOW_NAME]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$WORKFLOW_NAME]}
    export LCDB_OUTPUT_WORKFLOW=$output_path/results/$CAMPAIGN_NAME/$LCDB_WORKFLOW
else
    echo "Invalid workflow name: '$WORKFLOW_NAME'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

export LCDB_NUM_CONFIGS="$DEFAULT_CONFIG_NUM"
export LCDB_WORKFLOW_SEED="$DEFAULT_WORKFLOW_SEED"
export VAL_SEEDS="$DEFAULT_VAL_SEEDS"
export TEST_SEEDS="$DEFAULT_TEST_SEEDS"
export CAMPAIGN_NAME="$DEFAULT_CAMPAIGN_NAME"

# ********* Config Path Logic *********
INITIAL_CONFIG_FILE="$path_to_snellius/initial_configs/enforced_defaults/${WORKFLOW_NAME}.csv"

if [[ -f "$INITIAL_CONFIG_FILE" ]]; then
    export LCDB_INITIAL_CONFIGS="$INITIAL_CONFIG_FILE"
    echo "Using pre-defined initial configs: $LCDB_INITIAL_CONFIGS"
else
    export LCDB_INITIAL_CONFIGS=$LCDB_OUTPUT_WORKFLOW/initial_configs.csv
    echo "Creating initial configs: $LCDB_INITIAL_CONFIGS"
fi

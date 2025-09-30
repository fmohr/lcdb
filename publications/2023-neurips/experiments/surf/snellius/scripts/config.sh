#!/bin/bash
source ~/.bashrc
conda activate lcdb

# *********Loading Configuration*********

CONFIG_FILE="$path_to_snellius/scripts/config.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

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

# If WORKFLOW_NAME not already set by wrapper, read from YAML
if [[ -z "$WORKFLOW_NAME" ]]; then
    export WORKFLOW_NAME
    WORKFLOW_NAME=$(yq -r '.workflow_name' "$CONFIG_FILE")
fi

# check that the workflow name exists
if [[ -n "${mapping[$WORKFLOW_NAME]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$WORKFLOW_NAME]}
    export LCDB_OUTPUT_WORKFLOW=$output_path/results_test/$WORKFLOW_NAME/output/$LCDB_WORKFLOW
else
    echo "Invalid workflow name: '$WORKFLOW_NAME'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

export LCDB_NUM_CONFIGS=$(yq '.config_num' "$CONFIG_FILE")
export LCDB_WORKFLOW_SEED=$(yq '.workflow_seed' "$CONFIG_FILE")

# Respect externally-provided DESIRED_MEMORY_GB (per-bin) if set; else fallback to YAML
if [[ -z "${DESIRED_MEMORY_GB:-}" ]]; then
    export DESIRED_MEMORY_GB
    DESIRED_MEMORY_GB=$(yq -r '.desired_memory_GB' "$CONFIG_FILE")
fi

export VAL_SEEDS="$(yq -r '.val_seeds[]' "$CONFIG_FILE" | xargs)"
export TEST_SEEDS="$(yq -r '.test_seeds[]' "$CONFIG_FILE" | xargs)"
export CAMPAIGN_NAME=$(yq -r '.campaign_name' "$CONFIG_FILE" | tr -d '"') 

# *********Memory Calculations*********
# cluster profile
# genoa
export NODES=2
CPUS_PER_TASK=192
MEMORY_PER_NODE_GB=336

# rome example:
# export NODES=2
# CPUS_PER_TASK=128
# MEMORY_PER_NODE_GB=224

TOTAL_MEMORY_GB=$((MEMORY_PER_NODE_GB * NODES))

# Calculate desired cores with float math (supports decimal memory bins)
CORES_FLOAT=$(echo "scale=4; $TOTAL_MEMORY_GB / $DESIRED_MEMORY_GB" | bc -l)
DESIRED_CORES=$(printf "%.0f" "$CORES_FLOAT")

# Cap at total available cores across all nodes
MAX_CORES=$((CPUS_PER_TASK * NODES))
if (( DESIRED_CORES > MAX_CORES )); then
    DESIRED_CORES=$MAX_CORES
fi
export DESIRED_CORES

# Add CPUS_PER_CONFIG logic
CPUS_PER_CONFIG=$((MAX_CORES / DESIRED_CORES))
if (( CPUS_PER_CONFIG < 1 )); then
    CPUS_PER_CONFIG=1
fi
if (( CPUS_PER_CONFIG > 16 )); then
    CPUS_PER_CONFIG=16
fi
export CPUS_PER_CONFIG

# print desired cores and cpus per config
# echo "Using $DESIRED_CORES cores and $CPUS_PER_CONFIG CPUs per config."

# memory per core based on actual number of cores allocated
MEMORY_PER_CORE_GB=$(echo "$TOTAL_MEMORY_GB / $DESIRED_CORES" | bc -l)
LCDB_WORKFLOW_MEMORY_LIMIT_GB=$(printf "%.0f" "$MEMORY_PER_CORE_GB")
LCDB_WORKFLOW_MEMORY_LIMIT_MB=$((LCDB_WORKFLOW_MEMORY_LIMIT_GB * 1024))

export LCDB_WORKFLOW_MEMORY_LIMIT=$(echo "$LCDB_WORKFLOW_MEMORY_LIMIT_MB" | xargs)
export LCDB_WORKFLOW_MEMORY_LIMIT_GB

# write the memory per core back to the YAML file (for reproducibility/logging)
yq ".desired_memory_GB = $LCDB_WORKFLOW_MEMORY_LIMIT_GB" scripts/config.yaml -i -y

echo "Using $DESIRED_CORES cores across $NODES nodes, $CPUS_PER_CONFIG CPUs per config, and $LCDB_WORKFLOW_MEMORY_LIMIT_GB GB of memory per core."
echo "The updated memory usage has been saved to your config.yaml file."
# *********Memory Calculations*********

# ********* Datasets Loading (backward compatible) *********
# If LCDB_OPENML_ID_ARRAY is already exported (e.g., per bin), don't override.
if [[ -z "${LCDB_OPENML_ID_ARRAY+x}" || "${#LCDB_OPENML_ID_ARRAY[@]}" -eq 0 ]]; then
    declare -a values
    if [[ -n "${WORKFLOW_DATASET_CSV:-}" && -f "$WORKFLOW_DATASET_CSV" ]]; then
        # 2-col CSV (id,bin) — use first column
        while IFS=, read -r c0 c1 _; do
            [[ -n "$c0" ]] && values+=("$(echo "$c0" | xargs)")
        done < "$WORKFLOW_DATASET_CSV"
    else
        # Legacy single-column CSV
        while IFS= read -r line || [[ -n "$line" ]]; do
            values+=("$line")
        done < "$path_to_snellius/datasets_to_test.csv"
    fi
    export LCDB_OPENML_ID_ARRAY=(${values[@]})
fi

# ********* Config Path Logic *********
INITIAL_CONFIG_FILE="$path_to_snellius/randomized_preprocessor_configs/${WORKFLOW_NAME}.csv"

if [[ -f "$INITIAL_CONFIG_FILE" ]]; then
    export LCDB_INITIAL_CONFIGS="$INITIAL_CONFIG_FILE"
    echo "Using pre-defined initial configs: $LCDB_INITIAL_CONFIGS"
else
    export LCDB_INITIAL_CONFIGS=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/initial_configs.csv
    echo "Creating initial configs: $LCDB_INITIAL_CONFIGS"
fi

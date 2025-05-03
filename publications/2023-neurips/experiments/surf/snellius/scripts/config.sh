#!/bin/bash
source ~/.bashrc
conda activate lcdb

# *********Loading Configuration*********

CONFIG_FILE="$path_to_snellius/scripts/config.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi


# Workflow mapping
declare -A mapping
mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
)

# Read the workflow name from config.yaml and remove any extra quotes or spaces
export WORKFLOW_NAME=$(yq -r '.workflow_name' "$CONFIG_FILE")

# Ensure that the workflow name exists in the mapping
if [[ -n "${mapping[$WORKFLOW_NAME]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$WORKFLOW_NAME]}
    export LCDB_OUTPUT_WORKFLOW=$output_path/results/$WORKFLOW_NAME/output/$LCDB_WORKFLOW
else
    echo "Invalid workflow name: '$WORKFLOW_NAME'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

export LCDB_NUM_CONFIGS=$(yq '.config_num' "$CONFIG_FILE")
export LCDB_WORKFLOW_SEED=$(yq '.workflow_seed' "$CONFIG_FILE")
export DESIRED_MEMORY_GB=$(yq -r '.desired_memory_GB' "$CONFIG_FILE")
# export VAL_SEEDS=($(yq -r '.val_seeds[]' "$CONFIG_FILE"))
# export TEST_SEEDS=($(yq -r '.test_seeds[]' "$CONFIG_FILE"))
# passing this with export as string to the run.sh script
export VAL_SEEDS="$(yq -r '.val_seeds[]' "$CONFIG_FILE" | xargs)"
export TEST_SEEDS="$(yq -r '.test_seeds[]' "$CONFIG_FILE" | xargs)"


export CAMPAIGN_NAME=$(yq -r '.campaign_name' "$CONFIG_FILE" | tr -d '"') 

# *********Loading Configuration*********


# *********Memory Calculations*********
# Number of nodes
export NODES=1
CPUS_PER_TASK=192
MEMORY_PER_NODE_GB=336

# Calculate the number of cores based on the desired memory (in GB)
TOTAL_MEMORY_GB=$((MEMORY_PER_NODE_GB * NODES))
TOTAL_MEMORY_MB=$((TOTAL_MEMORY_GB * 1024))

# Calculate the number of cores to use based on the desired memory
DESIRED_CORES=$((TOTAL_MEMORY_GB / DESIRED_MEMORY_GB))

# Ensure that the number of cores does not exceed the available cores (CPUS_PER_TASK)
export DESIRED_CORES=$((DESIRED_CORES > CPUS_PER_TASK ? CPUS_PER_TASK : DESIRED_CORES))

# Memory per core based on the desired memory and number of cores
MEMORY_PER_CORE_GB=$((TOTAL_MEMORY_GB / DESIRED_CORES))
MEMORY_PER_CORE_MB=$((MEMORY_PER_CORE_GB * 1024))
export LCDB_WORKFLOW_MEMORY_LIMIT=$MEMORY_PER_CORE_MB
export LCDB_WORKFLOW_MEMORY_LIMIT_GB=$MEMORY_PER_CORE_GB

# Write the memory per core back to the YAML file
yq ".desired_memory_GB = $LCDB_WORKFLOW_MEMORY_LIMIT_GB" scripts/config.yaml -i -y
echo "Using $DESIRED_CORES cores and $LCDB_WORKFLOW_MEMORY_LIMIT_GB GB of memory per core."
echo "The updated memory usage has been saved to your config.yaml file."
# *********Memory Calculations*********

# Read the CSV file and load into an array
declare -a values
while IFS= read -r line || [[ -n "$line" ]]; do
    values+=("$line")
done < "$path_to_snellius/datasets_to_test.csv"
export LCDB_OPENML_ID_ARRAY=(${values[@]})


export LCDB_INITIAL_CONFIGS=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/initial_configs.csv
# export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/$LCDB_OPENML_ID
# export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED

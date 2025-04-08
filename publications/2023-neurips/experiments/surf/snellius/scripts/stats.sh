#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home/jvanrijn/workspace/lcdb/publications/2023-neurips/experiments/surf/snellius/logs/out/statistics.out
#SBATCH --error=/home/jvanrijn/workspace/lcdb/publications/2023-neurips/experiments/surf/snellius/logs/err/statistics.err
#SBATCH --cpus-per-task=32
#SBATCH --job-name=get_statistics
#SBATCH --threads-per-core=1


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
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
)

# Load configuration
CONFIG_FILE="$path_to_snellius/scripts/config.yaml"


if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read the workflow name from config.yaml and remove any extra quotes or spaces
WORKFLOW_NAME=$(yq -r '.workflow_name' "$CONFIG_FILE")

# Ensure that the workflow name exists in the mapping
if [[ -n "${mapping[$WORKFLOW_NAME]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$WORKFLOW_NAME]}
else
    echo "Invalid workflow name: '$WORKFLOW_NAME'"
    exit 1
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Read memory and core configuration
DESIRED_MEMORY_GB=$(yq -r '.desired_memory_GB' "$CONFIG_FILE")
val_seeds=($(yq -r '.val_seeds[]' "$CONFIG_FILE"))
test_seeds=($(yq -r '.test_seeds[]' "$CONFIG_FILE"))
num_configs=$(yq -r '.config_num' "$CONFIG_FILE")

campaign_name="data_probing-$DESIRED_MEMORY_GB"

echo "Campaign name: $campaign_name"

output_dir="../../../statistics-test/"

srun lcdb stats \
    --campaign-name $campaign_name \
    --output-dir $output_dir \
    --workflow-class $LCDB_WORKFLOW \
    --num-configs $num_configs 



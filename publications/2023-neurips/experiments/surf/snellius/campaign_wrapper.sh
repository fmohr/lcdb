#!/bin/bash
set -xe

# get the workflow
declare -A mapping
mapping=(
    ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    ["randomforest"]="lcdb.workflow.sklearn.RandomForestWorkflow"
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
    ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
)

# Retrieve the value based on the key
if [[ -n "${mapping[$1]}" ]]; then
    export LCDB_WORKFLOW=${mapping[$1]}
    export LCDB_OUTPUT_WORKFLOW=$PWD/$1/output/$LCDB_WORKFLOW

else
    echo "Invalid algorithm: '$1'"
fi

echo "'$LCDB_WORKFLOW' and '$LCDB_OUTPUT_WORKFLOW'"

# Redirect output and error to the appropriate log file
log_dir="$PWD/$1"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper_campaign.log") 2>&1

# Define the range of seeds
val_seeds=(0 1 2 3 4)
test_seeds=(0 1 2 3 4)

source config.sh

# Number of nodes
# TODO: Not yet supported but should investigate
NODES=1

for val_seed in "${val_seeds[@]}"; do
    for test_seed in "${test_seeds[@]}"; do
        export LCDB_VALID_SEED=$val_seed
        export LCDB_TEST_SEED=$test_seed

        # export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED/"results.csv.gz"

        # Get the size of the array of datasets
        array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))

        SBATCH_SCRIPT='run_campaign.sh'

        # Jobname is assigned the name of the workflow 
        jobname=''$1'_campaign'

        echo ""$LCDB_OUTPUT_RUN""
        # Submit the job using sbatch
        sbatch --export=all --job-name=$jobname --array=0-$array_size "$SBATCH_SCRIPT"
    done
done
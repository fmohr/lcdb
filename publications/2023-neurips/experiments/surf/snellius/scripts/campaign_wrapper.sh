#!/bin/bash
# set -xe

    # ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    # ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    # ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    # ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
# Get the workflow mappings
declare -A mapping
mapping=(
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
)

# campaign wrapper logging
log_dir="$PWD/logs"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper_campaign.log") 2>&1

val_seeds=(0)
test_seeds=(0)

source scripts/config.sh

for key in "${!mapping[@]}"; do
    export workflow=${mapping[$key]}
    export LCDB_WORKFLOW=$workflow
    export LCDB_OUTPUT_WORKFLOW="$PWD/$key/output/$LCDB_WORKFLOW"

    echo "Processing workflow '$LCDB_WORKFLOW' with output path '$LCDB_OUTPUT_WORKFLOW'"


    for val_seed in "${val_seeds[@]}"; do
        for test_seed in "${test_seeds[@]}"; do
            export LCDB_VALID_SEED=$val_seed
            export LCDB_TEST_SEED=$test_seed
            array_size=$((${#LCDB_OPENML_ID_ARRAY[@]} - 1))
            SBATCH_SCRIPT='scripts/campaign.sh'
            jobname="campaigns"

            sbatch --export=ALL, --array=0-$array_size, \
            --job-name=$jobname \
            --output=logs/%x/logs/out/%x_workflow-${key}_openml_idx-%a_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}.log \
            --error=logs/%x/logs/err/%x_workflow-${key}_openml_idx-%a_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}.err \
            "$WRAPPER_SCRIPT""$SBATCH_SCRIPT"
        done
    done
done

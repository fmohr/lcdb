#!/bin/bash

# Workflow mappings
    # ["libsvm"]="lcdb.workflow.sklearn.LibSVMWorkflow"
    # ["knn"]="lcdb.workflow.sklearn.KNNWorkflow"
    # ["xgboost"]="lcdb.workflow.xgboost.XGBoostWorkflow"
    # ["treesensemble"]="lcdb.workflow.sklearn.TreesEnsembleWorkflow"
declare -A mapping=(
    ["liblinear"]="lcdb.workflow.sklearn.LibLinearWorkflow"
)

# add memory limit (input parameter) to the workflow
export MEM=$1

log_dir="$PWD/logs"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/wrapper_campaign.log") 2>&1

# val_seeds=(0 1 2 3 4)
# test_seeds=(0 1 2 3 4)
val_seeds=(0)
test_seeds=(0)

source scripts/config.sh

workflow_seed=$LCDB_WORKFLOW_SEED

for key in "${!mapping[@]}"; do
    export workflow=${mapping[$key]}
    export LCDB_WORKFLOW=$workflow
    export LCDB_OUTPUT_WORKFLOW="$PWD/$key/output/$LCDB_WORKFLOW"

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
    sbatch --export=ALL --job-name="campaigns_${key}" \
        --output=logs/%x/logs/out/%x_workflow-${key}.log \
        --error=logs/%x/logs/err/%x_workflow-${key}.err \
        scripts/campaign.sh "${result_files[@]}"
done

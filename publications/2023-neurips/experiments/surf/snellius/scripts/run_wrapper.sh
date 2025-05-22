#!/bin/bash
# set -xe
source ~/.bashrc
conda activate lcdb

export path_to_snellius=$(pwd)
export output_path="/gpfs/nvme1/0/prjs1064/LCDB2"

# Load config for additional settings
source "$path_to_snellius/scripts/config.sh"

log_dir="$output_path/logs/$WORKFLOW_NAME-$LCDB_WORKFLOW_MEMORY_LIMIT_GB"
echo "Log directory: $log_dir"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/run_wrapper.log") 2>&1

source "$path_to_snellius/scripts/config.sh"

# checking if the status file does not exist then add to remaining_ids
remaining_ids=()
for openml_id in "${LCDB_OPENML_ID_ARRAY[@]}"; do
    export LCDB_OPENML_ID=$openml_id
    status="submitted"
    LCDB_WORKFOW_DIR=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB
    checkpoint_dir="$LCDB_WORKFOW_DIR/exp-checkpoints/$CAMPAIGN_NAME"
    export STATUS_FILE_BASE="$checkpoint_dir/openmlID-$LCDB_OPENML_ID.$LCDB_WORKFLOW"
    STATUS_FILE="$STATUS_FILE_BASE.$status"

    mkdir -p "$checkpoint_dir"

    if [ -f "$STATUS_FILE" ]; then
        echo "File $STATUS_FILE already exists. Skipping this configuration."
        continue
    else
        echo "Creating status file: $STATUS_FILE"
        touch "$STATUS_FILE"
        remaining_ids+=("$openml_id")
    fi
done

# copy remaining_ids to LCDB_OPENML_ID_ARRAY
export LCDB_OPENML_ID_ARRAY=("${remaining_ids[@]}")

if [ ${#LCDB_OPENML_ID_ARRAY[@]} -eq 0 ]; then
    echo "No OpenML IDs to process. Exiting."
    exit 1
fi

# Submit create.sh as a Slurm job and force next jobs to wait
create_job_id=$(sbatch --export=ALL \
                        --output=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/create_datasets.log \
                        --error=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/create_datasets.err \
                        --chdir=${output_path} \
                        --parsable \
                        scripts/create.sh)
echo "Submitted create.sh with Job ID: $create_job_id"


script="$path_to_snellius/scripts/run.sh"
jobname="$CAMPAIGN_NAME-m_$LCDB_WORKFLOW_MEMORY_LIMIT_GB-wc_$WORKFLOW_NAME"

LCDB_OPENML_ARRAY_STRING=$(IFS=,; echo "${LCDB_OPENML_ID_ARRAY[*]}")

# TODO: Get number of nodes and cores based on statistics (per workflow-dataset tuple)

# Submit run.sh with dependency on create.sh
sbatch --export=ALL --job-name=$jobname \
    --exclusive \
    --dependency=afterok:$create_job_id \
    --ntasks=$DESIRED_CORES \
    --array=$LCDB_OPENML_ARRAY_STRING \
    --mem=0 \
    --nodes=$NODES \
    --chdir=$output_path \
    --output=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/out.log \
    --error=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/err.log \
    "$script"

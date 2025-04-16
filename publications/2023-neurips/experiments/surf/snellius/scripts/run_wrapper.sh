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

# Submit create.sh as a Slurm job and force next jobs to wait
create_job_id=$(sbatch --export=ALL \
                        --output=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/create_datasets.log \
                        --error=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/create_datasets.err \
                        --chdir=${output_path} \
                        --parsable \
                        scripts/create.sh)
echo "Submitted create.sh with Job ID: $create_job_id"

for val_seed in "${VAL_SEEDS[@]}"; do
    for test_seed in "${TEST_SEEDS[@]}"; do
        for openml_id in "${LCDB_OPENML_ID_ARRAY[@]}"; do
            export LCDB_OPENML_ID=$openml_id
            export LCDB_VALID_SEED=$val_seed
            export LCDB_TEST_SEED=$test_seed
           
            status="submitted"
            source "$path_to_snellius/scripts/config.sh"
            export STATUS_FILE_BASE="$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME/$LCDB_WORKFLOW-$LCDB_OPENML_ID-$LCDB_WORKFLOW_SEED-$LCDB_TEST_SEED-$LCDB_VALID_SEED"
            STATUS_FILE="$STATUS_FILE_BASE.$status"
            # check if directory exists otherwise create it
            mkdir -p "$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME"

            # if file exists continue loop, else create it
            if [ -f "$STATUS_FILE" ]; then
                echo "File $STATUS_FILE already exists. Skipping this configuration."
                continue
            else
                echo "Creating status file: $STATUS_FILE"
                touch "$STATUS_FILE"
            fi
            
            script="$path_to_snellius/scripts/run.sh"
            jobname="$CAMPAIGN_NAME-m_$LCDB_WORKFLOW_MEMORY_LIMIT_GB-wc_$WORKFLOW_NAME-o_$LCDB_OPENML_ID-ws_$LCDB_WORKFLOW_SEED-vs_$LCDB_VALID_SEED-ts_$LCDB_TEST_SEED"

            # TODO: Get number of nodes and cores based on statistics (per workflow-dataset tuple)

            # Submit run.sh with dependency on create.sh
            sbatch --export=all --job-name=$jobname \
                --exclusive \
                --dependency=afterok:$create_job_id \
                --ntasks=$DESIRED_CORES \
                --mem=0 \
                --nodes=$NODES \
                --chdir=$output_path \
                --output=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}.log \
                --error=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${val_seed}_test-${test_seed}.err \
                "$script"

        done

    done
done

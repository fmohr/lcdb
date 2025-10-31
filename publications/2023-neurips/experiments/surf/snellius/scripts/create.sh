#!/bin/bash
#SBATCH --job-name=create_datasets
#SBATCH --time=01:00:00

# Load Python Environment
source ~/.bashrc
conda activate lcdbgpu

# Create Configurations

# Only generate initial configs if they do not exist
if [[ ! -f "$LCDB_INITIAL_CONFIGS" ]]; then
    echo "Initial config file not found at $LCDB_INITIAL_CONFIGS."
    # # Atomic creation of campaign status file
    if (set -o noclobber; : > "$CAMPAIGN_STATUS_FILE") 2> /dev/null; then
        echo "File $CAMPAIGN_STATUS_FILE did not exist. Creating initial configs."
        echo "Creating $LCDB_NUM_CONFIGS configurations for $LCDB_WORKFLOW in $LCDB_INITIAL_CONFIGS"
        lcdb create -w "$LCDB_WORKFLOW" -n "$LCDB_NUM_CONFIGS" -o "$LCDB_INITIAL_CONFIGS"
    else
        echo "Campaign status file \"$CAMPAIGN_STATUS_FILE\" already exists. Skipping initial config creation."
    fi
else
    echo "Initial configs already exist at $LCDB_INITIAL_CONFIGS. Skipping generation."
fi


# Fetch Datasets
for LCDB_OPENML_ID in ${LCDB_OPENML_ID_ARRAY[@]}; do
    echo "Fetching dataset $LCDB_OPENML_ID..."
    lcdb fetch --task-id openml.$LCDB_OPENML_ID
    echo ""
done

#!/bin/bash
#SBATCH --job-name=create_datasets
#SBATCH --output=logs/out/create_datasets.log
#SBATCH --error=logs/err/create_datasets.err
#SBATCH --partition=genoa
#SBATCH --time=01:00:00  # Adjust as needed

# Load Python Environment
source /home/$USER/workspace/lcdb/publications/2023-neurips/build/activate-dhenv.sh

# Load Experiment Configuration
source scripts/config.sh

# Create Configurations
echo "Creating $LCDB_NUM_CONFIGS configurations for $LCDB_WORKFLOW in $LCDB_INITIAL_CONFIGS"
lcdb create -w $LCDB_WORKFLOW -n $LCDB_NUM_CONFIGS -o $LCDB_INITIAL_CONFIGS

# Fetch Datasets
for LCDB_OPENML_ID in ${LCDB_OPENML_ID_ARRAY[@]}; do
    echo "Fetching dataset $LCDB_OPENML_ID..."
    lcdb fetch --task-id openml.$LCDB_OPENML_ID
    echo ""
done

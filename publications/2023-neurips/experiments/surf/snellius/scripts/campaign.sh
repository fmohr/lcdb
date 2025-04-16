#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --time=48:00:00
#SBATCH --threads-per-core=1

source ~/.bashrc
conda activate lcdb

declare -a result_files=("$@")


# Load the .env file if it exists
if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
fi

# update token before uploading data
srun lcdb pcloud -p "$ENV_PATH" || { echo "Authentication failed in lcdb pcloud, exiting."; exit 1; }

# re-load the .env file if it exists
if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
fi


for file in "${result_files[@]}"; do
  srun lcdb add -c "$CAMPAIGN_NAME" -t "$PCLOUD_TOKEN" "$file" || true
done
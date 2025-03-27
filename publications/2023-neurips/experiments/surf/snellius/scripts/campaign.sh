#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=48:00:00
#SBATCH --threads-per-core=1

source ~/.bashrc
conda activate lcdb

declare -a result_files=("$@")



# Define the path to the .env file (e.g., one level up)
ENV_PATH="../../../.env"  # Adjust as needed

# Load the .env file if it exists
if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
fi


for file in "${result_files[@]}"; do
  srun lcdb add -c "$campaign_name" -t "$PCLOUD_TOKEN" "$file" || true
done
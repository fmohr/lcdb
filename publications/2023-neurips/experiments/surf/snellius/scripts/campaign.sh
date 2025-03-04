#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=48:00:00
#SBATCH --threads-per-core=1

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/$USER/workspace/lcdb/publications/2023-neurips/build/activate-dhenv.sh

declare -a result_files=("$@")
# srun lcdb add -c data_probing -t "e7E1H7ZiFqSZR6Rfh7L38FRhOdBuicF5BuFfPbX7" "${result_files[@]}"

# campaing_name="data_probing- and exported MEM parameter
campaign_name="data_probing-$MEM"


# Define the path to the .env file (e.g., one level up)
ENV_PATH="../../../.env"  # Adjust as needed

# Load the .env file if it exists
if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
fi


for file in "${result_files[@]}"; do
  srun lcdb add -c "$campaign_name" -t "$PCLOUD_TOKEN" "$file" || true
done
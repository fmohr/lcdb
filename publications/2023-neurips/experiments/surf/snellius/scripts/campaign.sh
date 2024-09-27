#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --threads-per-core=1

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/$USER/projects/lcdb/publications/2023-neurips/build/activate-dhenv.sh

declare -a values
while IFS= read -r line || [[ -n "$line" ]]; do
    values+=("$line")
done < "./datasets_to_test.csv"
export LCDB_OPENML_ID_ARRAY=(${values[@]})

# source config
LCDB_OPENML_ID=${LCDB_OPENML_ID_ARRAY[$SLURM_ARRAY_TASK_ID]}
echo ""$LCDB_OPENML_ID""

export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW/$LCDB_OPENML_ID
export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED


echo ""$LCDB_OUTPUT_RUN""

# Run the srun command with all the paths
srun lcdb add -c snellius $LCDB_OUTPUT_RUN/"results.csv.gz"
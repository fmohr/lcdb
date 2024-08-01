#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --threads-per-core=1
#SBATCH --output=%x/out/%x_%a.log
#SBATCH --error=%x/err/%x_%a.log

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/$USER/projects/lcdb/publications/2023-neurips/build/activate-dhenv.sh

echo $LCDB_OUTPUT_RUN

# Run experiment
srun lcdb add -c snellius $LCDB_OUTPUT_RUN

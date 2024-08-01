#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --threads-per-core=1
#SBATCH --output=%x/out/%x_%a.log
#SBATCH --error=%x/err/%x_%a.log

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/$USER/projects/lcdb/publications/2023-neurips/build/activate-dhenv.sh

#!!! CONFIGURATION - START
source config.sh

export timeout=3500
export NTOTRANKS=$(( $SLURM_JOB_NUM_NODES * $SLURM_CPUS_PER_TASK ))
export PLOT_TYPE='test'
#!!! CONFIGURATION - END

mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN

# Run experiment
srun -n ${NTOTRANKS} -N ${SLURM_JOB_NUM_NODES} \
     --cpus-per-task 1 \
     --threads-per-core 1 \
    lcdb run \
    --openml-id $LCDB_OPENML_ID \
    --workflow-class $LCDB_WORKFLOW \
    --monotonic \
    --max-evals $LCDB_NUM_CONFIGS \
    --timeout $timeout \
    --initial-configs $LCDB_INITIAL_CONFIGS \
    --timeout-on-fit 300 \
    --workflow-seed $LCDB_WORKFLOW_SEED \
    --workflow-memory-limit $SLURM_MEM_PER_CPU \
    --valid-seed $LCDB_VALID_SEED \
    --test-seed $LCDB_TEST_SEED \
    --evaluator mpicomm 

gzip --best results.csv 


# srun -n 1 lcdb plot \
#     --results-path $LCDB_OUTPUT_RUN \
#     --output-path $LCDB_OUTPUT_RUN \
#     --plot-type $PLOT_TYPE
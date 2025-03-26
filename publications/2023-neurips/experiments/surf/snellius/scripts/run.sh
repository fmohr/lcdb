#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --threads-per-core=1

# module load 2024
# module load OpenMPI/5.0.3-GCC-13.3.0 

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0 

source ~/.bashrc
conda activate lcdb

#!!! CONFIGURATION - START
source scripts/config.sh

export timeout=3500
export NTOTRANKS=$DESIRED_CORES
#!!! CONFIGURATION - END

mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN

# Run experiment
# Documenting arguments of srun
# https://slurm.schedmd.com/srun.html
# -n --ntasks: number of tasks/ranks to run globally
# -N --nodes: number of nodes
# therefore the number of tasks/node is n/N
srun -n ${NTOTRANKS} -N ${SLURM_JOB_NUM_NODES:-1} \
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
    --workflow-memory-limit $LCDB_WORKFLOW_MEMORY_LIMIT \
    --valid-seed $LCDB_VALID_SEED \
    --no-exception-on-unsuitable-preprocessor \
    --test-seed $LCDB_TEST_SEED \
    --log-level debug \
    --evaluator mpicomm \
    --epoch-schedule=power-2-0.25-0 
    
gzip --best results.csv 
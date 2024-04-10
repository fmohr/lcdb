#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=128
#SBATCH --threads-per-core=1
#SBATCH --job-name=knn
#SBATCH --output=out/knn_%a.log
#SBATCH --error=err/knn_%a.log

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/jvanrijn/projects/lcdb/publications/2023-neurips/build/activate-dhenv.sh

#!!! CONFIGURATION - START
source config.sh

export timeout=3500

export NRANKS_PER_NODE=128
export NTASKS=$SLURM_ARRAY_TASK_COUNT
export CORES_PER_NODE=$SLURM_CPUS_PER_TASK  
export NNODES=$SLURM_JOB_NUM_NODES
export NODES_PER_TASK=$(( $NNODES / $NTASKS ))
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export CORES_PER_TASK=$(( $CORES_PER_NODE / $NRANKS_PER_NODE))
export THREADS_PER_CORE=$SLURM_THREADS_PER_CORE
export OMP_NUM_THREADS=$(( $CORES_PER_TASK * $THREADS_PER_CORE ))
#!!! CONFIGURATION - END

mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN

echo "Initial configs '$LCDB_INITIAL_CONFIGS'"

# Run experiment
srun -n ${NTOTRANKS} -N ${NNODES} \
     --cpus-per-task $CORES_PER_TASK \
     --threads-per-core $THREADS_PER_CORE \
     lcdb run \
    --openml-id $LCDB_OPENML_ID \
    --workflow-class $LCDB_WORKFLOW \
    --monotonic \
    --max-evals $LCDB_NUM_CONFIGS \
    --timeout $timeout \
    --initial-configs $LCDB_INITIAL_CONFIGS \
    --timeout-on-fit 300 \
    --workflow-seed $LCDB_WORKFLOW_SEED \
    --valid-seed $LCDB_VALID_SEED \
    --test-seed $LCDB_TEST_SEED \
    --evaluator mpicomm

gzip --best results.csv

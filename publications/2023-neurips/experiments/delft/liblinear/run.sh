#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --job-name=liblinear
#SBATCH --output=out/%a.log
#SBATCH --error=err/%a.log
#SBATCH --array=0-1
module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/jvanrijn/projects/lcdb/publications/2023-neurips/build/activate-dhenv.sh

#!!! CONFIGURATION - START
source config.sh

export timeout=3500

export NDEPTH=2
export NRANKS_PER_NODE=16
export NNODES=4
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE))
export OMP_NUM_THREADS=$NDEPTH
#!!! CONFIGURATION - END

mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN

# Run experiment
# srun -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --envall \
srun -n $(( $NDEPTH * $NRANKS_PER_NODE )) -N $NDEPTH\
    /home/jvanrijn/miniconda3/envs/dhenv/bin/lcdb run \
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

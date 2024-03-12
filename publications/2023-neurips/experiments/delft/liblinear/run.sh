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
module load miniconda/3.11
module load openmpi/4.0.1

source ../../../build/activate-dhenv.sh

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
srun -n $(( $NDEPTH * $NRANKS_PER_NODE )) -N $NDEPTH ../../../build/dhenv/bin/lcdb run \
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

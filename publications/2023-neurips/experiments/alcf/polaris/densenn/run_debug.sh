#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:60:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
source ./config.sh

export timeout=3500

export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE + 1))
export OMP_NUM_THREADS=$NDEPTH
#!!! CONFIGURATION - END

export output_workflow_dir=output/$LCDB_WORKFLOW
export output_dataset_dir=$output_workflow_dir/$LCDB_OPENML_ID
export output_run_dir=$output_dataset_dir/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED
mkdir -p $output_run_dir
pushd $output_run_dir

mpiexec -n ${NTOTRANKS} -host ${RANKS_HOSTS} \
    --envall \
    ../set_affinity_gpu_polaris.sh lcdb run \
    --openml-id $LCDB_OPENML_ID \
    --workflow $LCDB_WORKFLOW \
    --monotonic \
    --max-evals $LCDB_NUM_CONFIGS \
    --timeout $timeout \
    --initial-configs configs.csv \
    --timeout-on-fit 300 \
    --workflow-seed $LCDB_WORKFLOW_SEED \
    --valid-seed $LCDB_VALID_SEED \
    --test-seed $LCDB_TEST_SEED \
    --evaluator mpicomm
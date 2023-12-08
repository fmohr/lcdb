#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:60:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=grand:home

set -e

cd ${PBS_O_WORKDIR}

source /lus/grand/projects/datascience/regele/polaris/lcdb/publications/2023-neurips/build/activate-dhenv.sh

export timeout=3500

#!!! CONFIGURATION - START
export NDEPTH=2
export NRANKS_PER_NODE=16
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE))
export OMP_NUM_THREADS=$NDEPTH
#!!! CONFIGURATION - END

export LCDB_WORKFLOW=lcdb.workflow.sklearn.ConstantWorkflow
export LCDB_OUTPUT_WORKFLOW=$PWD/output/$LCDB_WORKFLOW
export LCDB_OPENML_IDS=(3 6 11 12 14 15 16 18 22 23 28 29 31 32 37 38 44 50 54 151 182 188 300 307 458 469 554 1049 1050 1053 1063 1067 1068 1461 1462 1464 1468 1475 1478 1480 1485 1486 1487 1489 1494 1497 1501 1510 1590 4134 4534 4538 23381 23517 40499 40668 40670 40701 40923 40927 40975 40978 40979 40982 40983 40984 40996 41027)
export LCDB_WORKFLOW_SEED=0
export LCDB_VALID_SEEDS=(0 1 2 3 4)
export LCDB_TEST_SEEDS=(0 1 2 3 4)

# Loop
for LCDB_OPENML_ID in $LCDB_OPENML_IDS do

    echo "Running experiment for dataset $LCDB_OPENML_ID"

    mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} \
        --cpu-bind depth --envall ./test.sh

done

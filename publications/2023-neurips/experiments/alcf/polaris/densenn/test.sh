#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# Load Python Environment
source /lus/grand/projects/datascience/regele/polaris/lcdb/publications/2023-neurips/build/activate-dhenv.sh

# Load Experiment Configuration
source ./config.sh

# Create Configurations
pushd $LCDB_OUTPUT_WORKFLOW
lcdb test \
    --openml-id $LCDB_OPENML_ID \
    --workflow-class $LCDB_WORKFLOW \
    --monotonic \
    --valid-seed $LCDB_VALID_SEED \
    --test-seed $LCDB_TEST_SEED \
    --workflow-seed $LCDB_WORKFLOW_SEED \
    --timeout-on-fit -1 > test-output.json

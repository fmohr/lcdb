#!/bin/bash

set -xe

# Load Python Environment
source /lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/activate-dhenv.sh

# Load Experiment Configuration
source ./config.sh

# Create Configurations
pushd $LCDB_WORKFLOW
lcdb test \
    --openml-id $LCDB_OPENML_ID \
    --workflow-class $LCDB_WORKFLOW \
    --monotonic \
    --valid-seed $LCDB_VALID_SEED \
    --test-seed $LCDB_TEST_SEED \
    --workflow-seed $LCDB_WORKFLOW_SEED \
    --timeout-on-fit -1 > test-output.json
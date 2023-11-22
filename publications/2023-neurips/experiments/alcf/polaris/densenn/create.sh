#!/bin/bash

set -xe

# Load Python Environment
source /lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/activate-dhenv.sh

# Load Experiment Configuration
source ./config.sh

# Create Configurations
pushd $LCDB_OUTPUT_WORKFLOW
lcdb create -w $LCDB_WORKFLOW -n $LCDB_NUM_CONFIGS -o configs.csv
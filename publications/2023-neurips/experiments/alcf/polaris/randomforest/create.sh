#!/bin/bash

set -e

# Load Python Environment
source /lus/grand/projects/datascience/regele/polaris/lcdb/publications/2023-neurips/build/activate-dhenv.sh

# Load Experiment Configuration
source ./config.sh

# Create Configurations
mkdir -p $LCDB_OUTPUT_WORKFLOW
pushd $LCDB_OUTPUT_WORKFLOW
lcdb create -w $LCDB_WORKFLOW -n $LCDB_NUM_CONFIGS -o configs.csv

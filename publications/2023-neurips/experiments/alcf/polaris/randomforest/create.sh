#!/bin/bash

set -xe

# Load Python Environment
source /lus/grand/projects/datascience/regele/polaris/lcdb/publications/2023-neurips/build/activate-dhenv.sh

# Load Experiment Configuration
source config.sh

# Create Configurations
lcdb create -w $LCDB_WORKFLOW -n $LCDB_NUM_CONFIGS -o $LCDB_INITIAL_CONFIGS

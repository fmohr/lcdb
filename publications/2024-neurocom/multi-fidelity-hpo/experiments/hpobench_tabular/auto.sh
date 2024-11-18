#!/bin/bash

set -xe

# directories=("navalpropulsion" "parkinsonstelemonitoring" "proteinstructure" "slicelocalization")
directories=("parkinsonstelemonitoring" "proteinstructure" "slicelocalization")
# directories=("proteinstructure" "slicelocalization")
# directories=("slicelocalization")


for directory in ${directories[@]}; do
    pushd $directory
    ./sequential-dbo.sh
    # ./sequential-optuna-tpe.sh
    # ./sequential-random.sh
    popd
done
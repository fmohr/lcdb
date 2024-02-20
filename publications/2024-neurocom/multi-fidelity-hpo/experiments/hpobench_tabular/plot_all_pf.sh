#!/bin/bash

export TASKS=("navalpropulsion" "parkinsonstelemonitoring" "proteinstructure" "slicelocalization")

for EXP_TASK in ${TASKS[@]}; do
    echo "Plotting $EXP_TASK"
    export EXP_TASK=$EXP_TASK
    python plot_pf.py
done

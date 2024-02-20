#!/bin/bash

export TASKS=(554 151 1461 6 300 1497 3 40982 1494 31 14 16 1464)

for EXP_TASK in ${TASKS[@]}; do
    echo "Plotting $EXP_TASK"
    export EXP_TASK=$EXP_TASK
    python plot_traj.py
done

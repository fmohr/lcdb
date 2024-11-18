#!/bin/bash

set -x

export problem="dhb_navalpropulsion"
export max_evals=200
export pruning_strategy="HB"
# export random_states=(42)
# export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113)
export random_states=(3348747335)

exec_search () {
    export log_dir="output/$problem-TPE-$pruning_strategy-$max_evals-$random_state"
    mkdir -p $log_dir

    # Create database
    export OPTUNA_DB_DIR="$log_dir/optunadb"
    export OPTUNA_DB_HOST="localhost"
    initdb -D "$OPTUNA_DB_DIR"
    pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" start
    createdb hpo

    mpirun -np 1 python -m scalbo.exp --problem $problem \
        --search "OPT-TPE" \
        --max-evals $max_evals \
        --random-state $random_state \
        --log-dir $log_dir \
        --pruning-strategy $pruning_strategy \
        --max-steps 100


    dropdb hpo
    pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" stop
}

for random_state in ${random_states[@]}; do
   for i in {1..5}; do 
        exec_search && break || sleep 5;
    done
   sleep 1;
done

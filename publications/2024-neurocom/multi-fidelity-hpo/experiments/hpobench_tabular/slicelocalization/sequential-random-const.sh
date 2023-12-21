#!/bin/bash

set -e

export DEEPHYPER_BENCHMARK_TASK="slicelocalization"

export problem="dhexp.benchmark.hpobench_tabular"
export search="deephyper.search.hps.CBO"
export stopper="deephyper.stopper.ConstantStopper"
export max_evals=200
# export random_states=(1608637542)
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 
export stop_steps=(1 5 10 25 50 75 100)

exec_search () {
    export log_dir="output/$problem-RANDOM-$stopper-$stop_step-$max_evals-$random_state"
    mkdir -p $log_dir

    python -m dhexp.run --problem $problem \
        --search $search \
        --search-kwargs "{'log_dir': '$log_dir', 'surrogate_model': 'DUMMY', 'random_state': $random_state}" \
        --stopper $stopper \
        --stopper-kwargs "{'max_steps': 100, 'stop_step': $stop_step}" \
        --max-evals $max_evals
}

for stop_step in ${stop_steps[@]}; do

  echo "stop_step: $stop_step"

  for random_state in ${random_states[@]}; do
    for i in {1..5}; do 
        exec_search && break || sleep 5;
    done
    sleep 1;
  done
done



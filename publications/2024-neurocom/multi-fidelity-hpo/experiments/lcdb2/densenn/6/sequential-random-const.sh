#!/bin/bash

set -e

export DEEPHYPER_BENCHMARK_MAX_FIDELITY=100
export DEEPHYPER_BENCHMARK_FIDELITY="epoch"
export DEEPHYPER_BENCHMARK_CSV="/Users/romainegele/Documents/Research/LCDB/lcdb/publications/2023-neurips/experiments/alcf/polaris/densenn/output/lcdb.workflow.keras.DenseNNWorkflow/6/42-42-42/results.csv.gz"

export problem="dhexp.benchmark.lcdb_hpo"
export search="deephyper.search.hps.CBO"
export stopper="deephyper.stopper.ConstantStopper"
export max_evals=200
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 
# export stop_steps=(1 2 3 4 5 10 15 20 25 50 75 100)
export stop_steps=(6 7 8 9 11 12 13 14 16 17 18 19 21 22 23 24 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99)

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
    echo "random_state: $random_state"
    for i in {1..5}; do 
        exec_search && break || sleep 5;
    done
    sleep 0;
  done
done



#!/bin/bash

set -e

export DEEPHYPER_BENCHMARK_MAX_FIDELITY=100
export DEEPHYPER_BENCHMARK_FIDELITY="epoch"
export DEEPHYPER_BENCHMARK_CSV="/Users/romainegele/Documents/Research/LCDB/lcdb/publications/2023-neurips/experiments/alcf/polaris/densenn/output/lcdb.workflow.keras.DenseNNWorkflow/554/42-42-42/results.csv.gz"

export problem="dhexp.benchmark.lcdb_hpo"
export search="deephyper.search.hps.CBO"
export stopper="deephyper.stopper.SuccessiveHalvingStopper"
export max_evals=200
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 
export reduction_factors=(1.19 1.41 2 4 8 16 32 64)

exec_search () {
    export log_dir="output/$problem-RANDOM-$stopper-$reduction_factor-$max_evals-$random_state"
    mkdir -p $log_dir

    python -m dhexp.run --problem $problem \
        --search $search \
        --search-kwargs "{'log_dir': '$log_dir', 'surrogate_model': 'DUMMY', 'random_state': $random_state}" \
        --stopper $stopper \
        --stopper-kwargs "{'max_steps': 100, 'min_steps': 1, 'reduction_factor': $reduction_factor}" \
        --max-evals $max_evals \
        --verbose
}

for reduction_factor in ${reduction_factors[@]}; do

  echo "reduction_factor: $reduction_factor"

  for random_state in ${random_states[@]}; do
    echo "random_state: $random_state"
    for i in {1..5}; do 
        exec_search && break || sleep 5;
    done
    sleep 1;
  done
done

#!/bin/bash

set -e

export DEEPHYPER_BENCHMARK_MAX_FIDELITY=100
export DEEPHYPER_BENCHMARK_FIDELITY="epoch"
export DEEPHYPER_BENCHMARK_CSV="/Users/romainegele/Documents/Research/LCDB/lcdb/publications/2023-neurips/experiments/alcf/polaris/densenn/output/lcdb.workflow.keras.DenseNNWorkflow/554/42-42-42/results.csv.gz"

export problem="dhexp.benchmark.lcdb_hpo"
export search="deephyper.search.hps.CBO"
export stopper="deephyper_benchmark.stopper.lcpfn.LCPFNStopper"
export max_evals=200
# export random_states=(1608637542)
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 
export prob_promotions=(0.5 0.95 0.7 0.8 0.9)

export PYTHONPATH="/Users/romainegele/Documents/Research/lcpfn:$PYTHONPATH"

exec_search () {
    export log_dir="output/$problem-RANDOM-$stopper-$prob_promotion-$max_evals-$random_state"
    mkdir -p $log_dir

    python -m dhexp.run --problem $problem \
        --search $search \
        --search-kwargs "{'log_dir': '$log_dir', 'surrogate_model': 'DUMMY', 'random_state': $random_state}" \
        --stopper $stopper \
        --stopper-kwargs "{'max_steps': 100, 'prob_promotion': $prob_promotion}" \
        --max-evals $max_evals \
        --verbose
}

for prob_promotion in ${prob_promotions[@]}; do

  echo "prob_promotion: $prob_promotion"

  for random_state in ${random_states[@]}; do
    echo "random_state: $random_state"
    for i in {1..5}; do 
        exec_search && break || sleep 5;
    done
    sleep 1;
  done
done

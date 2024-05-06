#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=7000
#SBATCH --threads-per-core=1
#SBATCH --output=out/%x_%a.log
#SBATCH --error=err/%x_%a.log

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/jvanrijn/projects/lcdb/publications/2023-neurips/build/activate-dhenv.sh

#!!! CONFIGURATION - START
source config.sh

export timeout=3500

export NRANKS_PER_NODE=32
export CORES_PER_NODE=$SLURM_CPUS_PER_TASK
export CORES_PER_TASK=$(( $CORES_PER_NODE / $NRANKS_PER_NODE))
export THREADS_PER_CORE=$SLURM_THREADS_PER_CORE
export OMP_NUM_THREADS=$(( $CORES_PER_TASK * $THREADS_PER_CORE ))
export NNODES=$SLURM_JOB_NUM_NODES
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
#!!! CONFIGURATION - END
export PLOT_TYPE='test'

mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN


# Run experiment
    #  --tres-per-task=cpu:$CORES_PER_TASK \
    #  --threads-per-core $THREADS_PER_CORE \
srun -n ${NTOTRANKS} -N ${NNODES} \
     --cpus-per-task $CORES_PER_TASK \
     --threads-per-core $THREADS_PER_CORE \
    /home/jvanrijn/miniconda3/envs/dhenv/bin/lcdb run \
    --openml-id $LCDB_OPENML_ID \
    --workflow-class $LCDB_WORKFLOW \
    --monotonic \
    --max-evals $LCDB_NUM_CONFIGS \
    --timeout $timeout \
    --initial-configs $LCDB_INITIAL_CONFIGS \
    --timeout-on-fit 300 \
    --workflow-seed $LCDB_WORKFLOW_SEED \
    --valid-seed $LCDB_VALID_SEED \
    --test-seed $LCDB_TEST_SEED \
    --evaluator mpicomm 

    #  lcdb test \
    # --openml-id $LCDB_OPENML_ID \
    # --workflow-class $LCDB_WORKFLOW \
    # --monotonic \
    # --valid-seed $LCDB_VALID_SEED \
    # --test-seed $LCDB_TEST_SEED \
    # --workflow-seed $LCDB_WORKFLOW_SEED \
    # --timeout-on-fit -1 > test-output.json
gzip --best results.csv 


srun -n 1 lcdb plot \
    --results-path $LCDB_OUTPUT_RUN \
    --output-path $LCDB_OUTPUT_RUN \
    --plot-type $PLOT_TYPE
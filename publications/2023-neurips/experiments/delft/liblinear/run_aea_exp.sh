#!/bin/bash
#SBATCH --partition=general --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=12 --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=lcdb2
#SBATCH --output=lcdbB2.txt
#SBATCH --error=lcdbB2.txt

module use /opt/insy/modulefiles
module load miniconda/3.11
module load openmpi/4.0.1

source $HOME/.bashrc # this is to make sure that we have CONDA available, could be done in a different way
source $HOME/user_support/tom_viering/build/activate-dhenv.sh

#!!! CONFIGURATION - START
source config.sh

export timeout=3500

export NDEPTH=2
export NRANKS_PER_NODE=16
export NNODES=4
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE))
export OMP_NUM_THREADS=$NDEPTH
#!!! CONFIGURATION - END

mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN

# Run experiment
srun -n ${SLURM_NTASKS} -c ${SLURM_CPUS_PER_TASK} lcdb run \
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

gzip --best results.csv

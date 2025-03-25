#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --time=48:00:00
#SBATCH --threads-per-core=1

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0 

source ~/.bashrc
conda activate lcdb

#!!! CONFIGURATION - START
source scripts/config.sh

export timeout=3500
export NTOTRANKS=$DESIRED_CORES

#!!! CONFIGURATION - END

mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN

# Documenting arguments of srun
# https://slurm.schedmd.com/srun.html
# -n --ntasks: number of tasks/ranks to run globally
# -N --nodes: number of nodes
# therefore the number of tasks/node is n/N

# Run experiment
srun --ntasks ${NTOTRANKS} --nodes ${SLURM_JOB_NUM_NODES} \
     --cpus-per-task=1 \
     --threads-per-core=1 \
     --mem-per-cpu=2000 \
     --exclusive \
     printenv > env1.log 

srun --ntasks ${NTOTRANKS} --nodes ${SLURM_JOB_NUM_NODES} \
     --cpus-per-task=1 \
     --threads-per-core=1 \
     --exclusive \
     --mem=100000 \
     printenv > env2.log 


srun --ntasks ${NTOTRANKS} --nodes ${SLURM_JOB_NUM_NODES} \
     --exclusive \
     --mem=100000 \
     printenv > env3.log 

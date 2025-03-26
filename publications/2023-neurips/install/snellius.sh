#!/bin/bash

# Generic installation script for DeepHyper on Surf Snellius.

# save starting directory
START_DIR=$(pwd)

# Module that provide conda (miniconda or else)
source ~/.bashrc
# Load modules available on the current system
module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0 

conda create -n lcdb python=3.11
conda activate lcdb
pip install --upgrade pip
pip install yq
python --version

export MPICC=$(which mpicc)
which mpicc
pip install mpi4py --no-cache-dir

cd /home/jvanrijn/miniconda3/envs/lcdb/compiler_compat
rm -f ld
ln -s /usr/bin/ld ld

pip install mpi4py --no-cache-dir

cd /home/jvanrijn/miniconda3/envs/lcdb/compiler_compat
rm -f ld
ln -s ../bin/x86_64-conda-linux-gnu-ld ld

# Return to the original directory
cd "$START_DIR"

# Install LCDB Package
pip install -e .
#!/bin/bash
# save starting directory
START_DIR=$(pwd)
env_name="lcdbgpu"

source ~/.bashrc
# Load modules available on the current system
module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0 

# Create a new conda environment
conda create -n $env_name python=3.11
conda activate $env_name

# Install the required packages
pip install --upgrade pip
pip install yq
pip install python-dotenv
pip install pyyaml

# install mpi4py
export MPICC=$(which mpicc)
which mpicc
pip install mpi4py --no-cache-dir

cd "/home/$USER/miniconda3/envs/$env_name/compiler_compat"
rm -f ld
ln -s /usr/bin/ld ld

pip install mpi4py --no-cache-dir

cd "/home/$USER/miniconda3/envs/$env_name/compiler_compat"
rm -f ld
ln -s ../bin/x86_64-conda-linux-gnu-ld ld

# return to the original directory
cd "$START_DIR"

# Install LCDB Package
pip install -e .
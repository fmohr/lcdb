#!/bin/bash

# Generic installation script for DeepHyper on ALCF's Polaris.
# This script is meant to be run on the login node of the machine.
# It will install DeepHyper and its dependencies in the current directory.
# A good practice is to create a `build` folder and launch the script from there,
# e.g. from the root of the DeepHyper repository:
# $ mkdir build && cd build && ../install/alcf/polaris.sh
# The script will also create a file named `activate-dhenv.sh` that will
# Setup the environment each time it is sourced `source activate-dhenv.sh`.

# set -xe
# mkdir build
cd build

# Load modules available on the current system
module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
# Module that provide conda (miniconda or else)
source ~/.bashrc

# conda create -n dhenv python=3.11
conda activate dhenv
pip install --upgrade pip
python --version

# For mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
export MPICC=$(which mpicc) 
which mpicc
pip install mpi4py --no-cache-dir
which mpicc

# Install the DeepHyper's Python package
git clone -b develop git@github.com:deephyper/deephyper.git
pip install -e "deephyper/[hps,mpi]"

# Install LCDB Package
cd ..
ls
pip install -e "../"
ls
# cd build

# Create activation script
touch activate-dhenv.sh
echo "file generated"
echo "#!/bin/bash" >> activate-dhenv.sh

# Append modules loading and conda activation
echo "" >> activate-dhenv.sh
# echo "module load PrgEnv-gnu/8.3.3" >> activate-dhenv.sh
echo "source ~/.bashrc" >> activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh
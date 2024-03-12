#!/bin/bash

# Generic installation script for DeepHyper on ALCF's Polaris.
# This script is meant to be run on the login node of the machine.
# It will install DeepHyper and its dependencies in the current directory.
# A good practice is to create a `build` folder and launch the script from there,
# e.g. from the root of the DeepHyper repository:
# $ mkdir build && cd build && ../install/alcf/polaris.sh
# The script will also create a file named `activate-dhenv.sh` that will
# Setup the environment each time it is sourced `source activate-dhenv.sh`.

set -xe

# Load modules available on the current system
module load PrgEnv-gnu/8.3.3
module load conda/2023-10-04

# Copy the base conda environment
conda create -p dhenv --clone base -y
conda activate dhenv/
pip install --upgrade pip

# Install the DeepHyper's Python package
git clone -b develop git@github.com:deephyper/deephyper.git
pip install -e "deephyper/[hps,mpi]"

# For mpi4py
module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
MPICC=CC python setup.py install

# Install LCDB Package
pip install -e "../"

# Create activation script
touch activate-dhenv.sh
echo "#!/bin/bash" >> activate-dhenv.sh

# Append modules loading and conda activation
echo "" >> activate-dhenv.sh
echo "module load PrgEnv-gnu/8.3.3" >> activate-dhenv.sh
echo "module load conda/2023-10-04" >> activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh

#!/bin/bash

# Generic installation script for DeepHyper on Surf Snellius.

# set -xe
mkdir build
cd build

# Module that provide conda (miniconda or else)
source ~/.bashrc
# Load modules available on the current system
module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0 

conda create -n dhenv python=3.11
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

# Install LCDB Package
cd ..
ls
pip install -e "../"
ls
# cd build

# Create activation script
# touch activate-dhenv.sh
# echo "file generated"
# echo "#!/bin/bash" >> activate-dhenv.sh

# # Append modules loading and conda activation
# echo "" >> activate-dhenv.sh
# echo "source ~/.bashrc" >> activate-dhenv.sh
# echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh
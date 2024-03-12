#!/bin/bash

set -xe

# Load modules available on the current system
module load miniconda/3.11

# Copy the base conda environment
conda create -p dhenv --clone base -y
conda activate dhenv/
pip install --upgrade pip

# Install the DeepHyper's Python package
pip install "deephyper @ git+https://github.com/deephyper/deephyper@develop"

# For mpi4py
module load openmpi/4.0.1
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
export MPICC=$(which mpicc)
echo "Installing mpi4py with mpicc=$MPICC"
export LD_LIBRARY_PATH="/usr/lib64:$LD_LIBRARY_PATH"
python setup.py install
cd ..

# Install LCDB Package
pip install -e "../"

# Create activation script
touch activate-dhenv.sh
echo "#!/bin/bash" >> activate-dhenv.sh


# Append modules loading and conda activation
echo "" >> activate-dhenv.sh
echo "module load openmpi/4.0.1" >> activate-dhenv.sh
echo "module load miniconda/3.11" >> activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh

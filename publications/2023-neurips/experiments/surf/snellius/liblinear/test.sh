#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=7000
#SBATCH --threads-per-core=1
#SBATCH --output=out/%x_test_%a.log
#SBATCH --error=err/%x_test_%a.log

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source /home/jvanrijn/projects/lcdb/publications/2023-neurips/build/activate-dhenv.sh

# Load Experiment Configuration
#!!! CONFIGURATION - START
source ./config.sh

export LCDB_EVALUATION_MEMORY_LIMIT=0.5

#!!! CONFIGURATION - END


mkdir -p $LCDB_OUTPUT_RUN
pushd $LCDB_OUTPUT_RUN

parameters='{"C": 0.2204499866449845, "class_weight": "none", "dual": false, "fit_intercept": false, "intercept_scaling": 342.4446701827744, "loss": "squared_hinge", "max_iter": 7565, "multiclass": "ovo-scikit", "penalty": "l1", "pp@cat_encoder": "onehot", "pp@decomposition": "kernel_pca", "pp@featuregen": "poly", "pp@featureselector": "selectp", "pp@scaler": "std", "tol": 0.0002422222995655, "pp@kernel_pca_kernel": "linear", "pp@kernel_pca_n_components": 0.8724077638181675, "pp@poly_degree": 2, "pp@selectp_percentile": 96, "pp@std_with_std": false}'


lcdb test \
        --openml-id $LCDB_OPENML_ID \
        --workflow-class $LCDB_WORKFLOW \
        --monotonic \
        --parameters "$parameters" \
        --timeout-on-fit -1 > test-output.json

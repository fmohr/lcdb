#!/bin/bash
#SBATCH --job-name=lcdb_gpu
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=${LCDB_WORKFLOW_MEMORY_LIMIT}MB
#SBATCH --time=6:00:00
#SBATCH --threads-per-core=1
#SBATCH --output=logs/lcdb_gpu_%A_%a.out
#SBATCH --error=logs/lcdb_gpu_%A_%a.err
#SBATCH --array=0-$((${NUM_DATASETS:-1}))

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0

source ~/.bashrc
conda activate lcdbgpu

# CONFIGURATION
export timeout=-1
export NTOTRANKS=$DESIRED_CORES

IFS=' ' read -r -a VAL_SEEDS <<< "$VAL_SEEDS"
IFS=' ' read -r -a TEST_SEEDS <<< "$TEST_SEEDS"
IFS=',' read -r -a LCDB_OPENML_ID_ARRAY <<< "$LCDB_OPENML_ARRAY_STRING"
LCDB_OPENML_ID=${LCDB_OPENML_ID_ARRAY[$SLURM_ARRAY_TASK_ID]}

echo "Running experiment for OpenML ID: $LCDB_OPENML_ID"
echo "Validation seeds: ${VAL_SEEDS[*]}"
echo "Test seeds: ${TEST_SEEDS[*]}"

export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/$LCDB_OPENML_ID

for LCDB_VALID_SEED in "${VAL_SEEDS[@]}"; do
    for LCDB_TEST_SEED in "${TEST_SEEDS[@]}"; do
        export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED
        mkdir -p $LCDB_OUTPUT_RUN
        pushd $LCDB_OUTPUT_RUN

        status="started"
        export STATUS_FILE_BASE="$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME/$LCDB_WORKFLOW-$LCDB_OPENML_ID-$LCDB_WORKFLOW_SEED-$LCDB_TEST_SEED-$LCDB_VALID_SEED"
        STATUS_FILE="$STATUS_FILE_BASE.$status"

        mkdir -p "$(dirname "$STATUS_FILE")"

        if [ -f "$STATUS_FILE" ]; then
            echo "File $STATUS_FILE already exists. Skipping."
            continue
        else
            echo "Creating status file: $STATUS_FILE"
            touch "$STATUS_FILE"

            srun -n $DESIRED_CORES -N $NODES \
                --cpus-per-task=1 \
                --gpus=$MIG_COUNT \
                --threads-per-core=1 \
                --exclusive \
                --output=${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.log \
                --error=${output_path}/logs_test/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.err \
                lcdb run \
                    --campaign $CAMPAIGN_NAME \
                    --openml-id $LCDB_OPENML_ID \
                    --workflow-class $LCDB_WORKFLOW \
                    --monotonic \
                    --max-evals $LCDB_NUM_CONFIGS \
                    --timeout $timeout \
                    --initial-configs $LCDB_INITIAL_CONFIGS \
                    --timeout-on-fit 300 \
                    --workflow-seed $LCDB_WORKFLOW_SEED \
                    --workflow-memory-limit $LCDB_WORKFLOW_MEMORY_LIMIT \
                    --valid-seed $LCDB_VALID_SEED \
                    --no-exception-on-unsuitable-preprocessor \
                    --test-seed $LCDB_TEST_SEED \
                    --log-level debug \
                    --evaluator mpicomm \
                    --epoch-schedule=power-2-0.25-0
              
            gzip --best results.csv 
        fi
    done
done

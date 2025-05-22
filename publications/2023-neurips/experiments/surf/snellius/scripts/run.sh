#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --time=48:00:00
#SBATCH --threads-per-core=1

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0 

source ~/.bashrc
conda activate lcdb

#!!! CONFIGURATION - START
export timeout=3500
export NTOTRANKS=$DESIRED_CORES
#!!! CONFIGURATION - END

echo "Running experiment for OpenML ID: $SLURM_ARRAY_TASK_ID"
IFS=' ' read -r -a VAL_SEEDS <<< "$VAL_SEEDS"
IFS=' ' read -r -a TEST_SEEDS <<< "$TEST_SEEDS"
echo "Validation seeds: ${VAL_SEEDS}"
echo "Test seeds: ${TEST_SEEDS}"

for LCDB_VALID_SEED in "${VAL_SEEDS[@]}"; do
    for LCDB_TEST_SEED in "${TEST_SEEDS[@]}"; do
        # get the openmlid from the array
        export LCDB_OPENML_ID=$SLURM_ARRAY_TASK_ID
        export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW-$LCDB_WORKFLOW_MEMORY_LIMIT_GB/$LCDB_OPENML_ID
        export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED

        mkdir -p $LCDB_OUTPUT_RUN
        pushd $LCDB_OUTPUT_RUN

        # Creating the 'started' status file
        status="started"
        export STATUS_FILE_BASE="$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME/$LCDB_WORKFLOW-$LCDB_OPENML_ID-$LCDB_WORKFLOW_SEED-$LCDB_TEST_SEED-$LCDB_VALID_SEED"
        STATUS_FILE="$STATUS_FILE_BASE.$status"
        # check if directory exists otherwise create it
        mkdir -p "$LCDB_OUTPUT_RUN/exp-checkpoints/$CAMPAIGN_NAME"

        # if file exists continue loop, else create it
        if [ -f "$STATUS_FILE" ]; then
            echo "File $STATUS_FILE already exists. Skipping this configuration."
            continue
        else
            echo "Creating status file: $STATUS_FILE"
            touch "$STATUS_FILE"
            # Run experiment
            # Documenting arguments of srun
            # https://slurm.schedmd.com/srun.html
            # -n --ntasks: number of tasks/ranks to run globally
            # -N --nodes: number of nodes
            # therefore the number of tasks/node is n/N
            srun -n ${NTOTRANKS} -N ${SLURM_JOB_NUM_NODES:-1} \
                    --cpus-per-task 1 \
                    --threads-per-core 1 \
                    --exclusive \
                    --output=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/out/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.log \
                    --error=${output_path}/logs/${WORKFLOW_NAME}-${LCDB_WORKFLOW_MEMORY_LIMIT_GB}/err/openml_id-${LCDB_OPENML_ID}_workflow-${LCDB_WORKFLOW_SEED}_val-${LCDB_VALID_SEED}_test-${LCDB_TEST_SEED}.err \
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
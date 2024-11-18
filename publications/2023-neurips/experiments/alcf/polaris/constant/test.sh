#!/bin/bash

if [ $PMI_RANK -lt 25 ]
then

    export LCDB_VALID_SEED=$(( ${PMI_RANK} % 5 ))
    export LCDB_TEST_SEED=$(( ${PMI_RANK} / 5 % 5 ))

    export LCDB_OUTPUT_DATASET=$LCDB_OUTPUT_WORKFLOW/$LCDB_OPENML_ID
    export LCDB_OUTPUT_RUN=$LCDB_OUTPUT_DATASET/$LCDB_VALID_SEED-$LCDB_TEST_SEED-$LCDB_WORKFLOW_SEED

    echo "Output in $LCDB_OUTPUT_RUN"

    mkdir -p $LCDB_OUTPUT_RUN
    pushd $LCDB_OUTPUT_RUN

    # Run experiment
    lcdb test --openml-id $LCDB_OPENML_ID \
        --workflow-class $LCDB_WORKFLOW \
        --monotonic \
        --valid-seed $LCDB_VALID_SEED \
        --test-seed $LCDB_TEST_SEED \
        --workflow-seed $LCDB_WORKFLOW_SEED > output.json

    gzip --best output.json

    popd

fi

#!/bin/bash

set -xe

source ../../../build/activate-dhenv.sh

source ./config.sh

mkdir -p output/$LCDB_WORKFLOW

lcdb create -w $LCDB_WORKFLOW -n $LCDB_NUM_CONFIGS -o output/$LCDB_WORKFLOW/configs.csv
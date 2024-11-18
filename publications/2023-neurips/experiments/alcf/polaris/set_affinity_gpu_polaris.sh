#!/bin/bash

gpu=$((${PMI_LOCAL_RANK} % ${NGPUS_PER_NODE}))
export CUDA_VISIBLE_DEVICES=$gpu
echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}”
exec "$@"

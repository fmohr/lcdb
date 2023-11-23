#!/bin/bash -l

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

mpiexec -n ${NNODES} --ppn 1 --hostfile ${PBS_NODEFILE} nvidia-cuda-mps-control -d

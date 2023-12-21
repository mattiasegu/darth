#!/bin/bash

EXP_NAME=$1
CONFIG=$2
CHECKPOINT=$3
CFG_OPTIONS=${@:4}

# Make conda available
eval "$(conda shell.bash hook)"
# Activate a conda environment
conda activate darth

python tools/run/train.py \
    ${CONFIG} \
    --exp-name ${EXP_NAME} \
    --resume-from ${CHECKPOINT} \
    --cfg-options ${CFG_OPTIONS[@]}
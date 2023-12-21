#!/bin/bash

CONFIG=$1
GPUS=$2
PY_ARGS=${@:3}

python tools/run/train.py \
    ${CONFIG} \
    ${PY_ARGS}
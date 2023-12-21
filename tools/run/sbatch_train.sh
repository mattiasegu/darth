#!/bin/bash

PARTITION=$1  # gpu22
TIME=$2  # 10:00:00
GRES=$3  # gpu:a40:1:03 | a100
CPUS=$4  # 16
MEM_PER_CPU=$5 # 5000
JOB_NAME=$6  # train_qdtrack_dancetrack
CONFIG=$7  # configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_dancetrack.py
CFG_OPTIONS=${@:8}

mkdir -p errors
mkdir -p outputs

echo "Starting job ${JOB_NAME} from ${CONFIG} using --cfg-options ${CFG_OPTIONS[*]}" 

# if you want to exclude nodes, add --exclude=gpu16-a40-[06-07]

sbatch -p ${PARTITION} \
    -t ${TIME} \
    -c ${CPUS} \
    --mem-per-cpu ${MEM_PER_CPU} \
    --gres ${GRES} \
    --job-name=${JOB_NAME} \
    -e errors/%j.log \
    -o outputs/%j.log \
    tools/run/train.sh ${JOB_NAME} ${CONFIG} -- ${CFG_OPTIONS[@]}

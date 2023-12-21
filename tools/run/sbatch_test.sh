#!/bin/bash

PARTITION=$1  # gpu22
TIME=$2  # 10:00:00
GPUS_NAME=$3  # a40 | a100
GPUS=$4  # 2 | 1
CPUS=$5  # 16
MEM_PER_CPU=$6 # 5000
JOB_NAME=$7  # train_qdtrack_dancetrack
CONFIG=$8  # work_dirs/train_kd_qdtrack_dancetrack_frcnn_4e_lr_002_rpn_001_roi_001/20220930_151753/kd_qdtrack_faster-rcnn_r50_fpn_4e_dancetrack.py
CHECKPOINT=$9  # work_dirs/train_kd_qdtrack_dancetrack_frcnn_4e_lr_002_rpn_001_roi_001/20220930_151753/latest.pth
CFG_OPTIONS=${@:10}

mkdir -p errors
mkdir -p outputs

echo "Resuming job ${JOB_NAME} from ${CONFIG} and ${CHECKPOINT} using --cfg-options ${CFG_OPTIONS[*]}" 
sbatch -p ${PARTITION} \
    -t ${TIME} \
    -c ${CPUS} \
    --mem-per-cpu ${MEM_PER_CPU} \
    --gres gpu:${GPUS_NAME}:${GPUS} \
    --job-name=${JOB_NAME} \
    -e errors/%j.log \
    -o outputs/%j.log \
    tools/run/test.sh ${JOB_NAME} ${CONFIG} ${CHECKPOINT} ${CFG_OPTIONS[@]}

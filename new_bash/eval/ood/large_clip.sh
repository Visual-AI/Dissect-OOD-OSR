#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='large'
MODEL='clip'
# OOD_METHODS=("msp" "mls" "energy")
OOD_METHODS=("mls")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE")
DIRS=("")

ARPL_DIRS=""

GODIN_DIRS=("" "")
GODIN_ARPL_CRT_DIRS=""

for s in ${!STRATEGIES[@]}; do
    for i in ${!OOD_METHODS[@]}; do
        ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
        --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_DIRS} --resume_godin-dir=${GODIN_DIRS[$s]} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS} >> ${SAVE_DIR}ood_clip.out
    done
done
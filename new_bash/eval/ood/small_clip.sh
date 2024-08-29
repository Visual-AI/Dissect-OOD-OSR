#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='small'
MODEL='clip'
OOD_METHODS=("msp" "mls" "energy" "odin")

MAGs=(0.05)
# MAGs=(1.0 0.5 0.25 0.1 0.05 0.025 0.0125 0.00625 0.003175)

SAVE_DIR=/disk/work/hjwang/osrd/exp/

GODIN_DIRS=("")
GODIN_ARPL_CRT_DIRS=""

STRATEGIES=("CE")
DIRS=("")
ARPL_CRT_DIRS=""


for s in ${!STRATEGIES[@]}; do
    for i in ${!OOD_METHODS[@]}; do
      for m in ${!MAGs[@]}; do
        ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
        --magnitude=${MAGs[$m]}\
        --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_CRT_DIRS} --resume_godin-dir=${GODIN_DIRS[$s]} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS} >> ${SAVE_DIR}ood_cifar10_clip.out
      done
    done
done
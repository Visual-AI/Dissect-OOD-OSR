#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

MODEL='clip_ft'

DATASETS=('cub' 'scars' 'aircraft')
# DATASETS=('imagenet21k')
# DATASETS=('waterbird')

# OOD_METHODS=("msp" "mls" "energy")
OOD_METHODS=("mls")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE")

for d in ${!DATASETS[@]}; do
  for s in ${!STRATEGIES[@]}; do
      for i in ${!OOD_METHODS[@]}; do
        ${PYTHON} -m eval.osr_large --model=${MODEL} --in-dataset=${DATASETS[$d]} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} \
        --split_idx=0 --image_size 224 --batch_size=32 --feat_dim=2048 --transform='rand-augment' \
        --resume-dir='' --resume_crit-dir='' >> ${SAVE_DIR}osr_clip_ft.out
      done
  done
done
#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

MODEL='clip_lp'

DATASETS=('imagenet21k')
# DATASETS=('waterbird')

OOD_METHODS=("msp" "mls" "energy")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE")

for d in ${!DATASETS[@]}; do
  for s in ${!STRATEGIES[@]}; do
      for i in ${!OOD_METHODS[@]}; do
        ${PYTHON} -m eval.osr_large --model=${MODEL} --in-dataset=${DATASETS[$d]} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} \
        --split_idx=0 --image_size 224 --batch_size=32 --feat_dim=2048 --transform='rand-augment' \
        --resume-dir='' --resume_crit-dir='' >> ${SAVE_DIR}osr_clip_lp.out
      done
  done
done
#!/bin/bash
PYTHON="/home/hjwang/anaconda3/envs/py37/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

OOD_METHODS="mls odin energy react_mls react_odin react_energy"

DATACOMBS='small middle large'
MODEL='resnet18'
DIRS="CE ARPL OE CE ARPL OE CE ARPL OE /disk1/hjwang/osrd/arpl_models_imagenet"

# MODEL='vit_small'
# DATACOMBS='small middle'
# DIRS="/disk1/hjwang/osrd/arpl_models_imagenet"


for comb in $DATACOMBS; do
  for dir in $DIRS; do
    for method in $OOD_METHODS; do
      ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${method} --loss_strategy=${strategy} --val_comb=${comb} --resume-dir=${dir}
    done
  done
done
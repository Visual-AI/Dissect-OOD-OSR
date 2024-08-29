#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='large'
# MODEL='resnet50'
MODEL='dino_vit_s8'
OOD_METHODS=("msp" "mls" "energy" "odin" "react_mls" "react_odin" "react_energy" "sem" "gradnorm")
STRATEGIES=("CE")

DIRS=("/disk/work/hjwang/osrd/arpl_models_imagenet/imagenet_9_ARPLoss.pth" "/disk/work/hjwang/osrd/arpl_models_imagenet/imagenet_9_ARPLoss.pth")
ARPL_DIRS="/disk/work/hjwang/osrd/arpl_models_imagenet/imagenet_9_ARPLoss_criterion.pth"

SAVE_DIR=/disk/work/hjwang/osrd/exp/

for s in ${!STRATEGIES[@]}; do
  if [ -f "${DIRS[$s]}" ]
  then
      for i in ${!OOD_METHODS[@]}; do
        ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
        --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_DIRS} >> ${SAVE_DIR}ood_SSB_dinos8.out
      done
  else
    continue
  fi
done
#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='large'
MODEL='resnet50'
OOD_METHODS=("msp" "mls" "energy" "odin" "react_mls" "react_odin" "react_energy" "gradnorm")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE" "ARPL" "OE")
DIRS=("/disk/work/hjwang/osrd/arpl_models_imagenet/imagenet_9_ARPLoss.pth" "/disk/work/hjwang/osrd/arpl_models_imagenet/imagenet_9_ARPLoss.pth" "/disk/work/hjwang/osrd/models_imagenet/imagenet_9_oe.pth")
ARPL_DIRS="/disk/work/hjwang/osrd/arpl_models_imagenet/imagenet_9_ARPLoss_criterion.pth"

ARPL_DIRS=""

GODIN_DIRS=("" "")
GODIN_ARPL_CRT_DIRS=""

for s in ${!STRATEGIES[@]}; do
  if [ -f "${DIRS[$s]}" ]
  then
    for i in ${!OOD_METHODS[@]}; do
      ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
      --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_DIRS} --resume_godin-dir=${GODIN_DIRS[$s]} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS} >> ${SAVE_DIR}ood_SSB_yfcc.out
    done
  else
    continue
  fi
done

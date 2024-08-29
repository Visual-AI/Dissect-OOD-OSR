#!/bin/bash
PYTHON="/home/hjwang/anaconda3/envs/py37/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='small'
MODEL='resnet50'
OOD_METHODS=("msp" "mls" "energy" "odin" "godin" "sem" "gradnorm" "react_mls" "react_odin" "react_energy")

STRATEGIES=("OE")
DIRS=("/home/hjwang/osrd/logs/cifar-10_resnet50_OE_conv-default/bestpoint.pth.tar" "/home/hjwang/osrd/logs/cifar-10_resnet50_OE_wrn-oe/bestpoint.pth.tar")
ARPL_DIRS=""

GODIN_DIRS=""
GODIN_ARPL_CRT_DIRS=""

for d in ${!DIRS[@]}; do
  if [ -f "${DIRS[$d]}" ]
  then
    for i in ${!OOD_METHODS[@]}; do
      ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES} --val_comb=${DATACOMBS} \
      --resume-dir=${DIRS[$d]} --resume_crit-dir=${ARPL_DIRS} --resume_godin-dir=${GODIN_DIRS} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS}
    done
  else
    continue
  fi
done
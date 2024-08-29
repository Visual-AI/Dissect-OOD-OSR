#!/bin/bash
PYTHON="/home/hjwang/anaconda3/envs/py37/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='middle'
MODEL='resnet18'
OOD_METHODS=("msp" "mls" "energy" "odin" "godin" "sem" "gradnorm" "react_mls" "react_odin" "react_energy")

STRATEGIES=("CE" "ARPL_CS" "OE")
DIRS=("/home/hjwang/osrd/logs/cifar-100_resnet18_CE_conv-default/bestpoint.pth.tar" "/home/hjwang/osrd/logs/cifar-100_resnet18_ARPL_CS_conv-cifar10-default/bestpoint.pth.tar" "/home/hjwang/osrd/logs/cifar-100_resnet18_OE_conv-default/bestpoint.pth.tar")
ARPL_CRT_DIRS="/home/hjwang/osrd/logs/cifar-100_resnet18_ARPL_CS_conv-cifar10-default/criterion.pth.tar"

GODIN_DIRS=("/home/hjwang/osrd/logs/cifar-100_resnet18_CE_godin_conv-cifar10-default/bestpoint.pth.tar" "/home/hjwang/osrd/logs/cifar-100_resnet18_ARPL_CS_godin_conv-cifar10-default/bestpoint.pth.tar" "/home/hjwang/osrd/logs/cifar-100_resnet18_OE_godin_conv-default/bestpoint.pth.tar")
GODIN_ARPL_CRT_DIRS="/home/hjwang/osrd/logs/cifar-100_resnet18_ARPL_CS_godin_conv-cifar10-default/criterion.pth.tar"

for s in ${!STRATEGIES[@]}; do
  if [ -f "${DIRS[$s]}" ]
  then
    for i in ${!OOD_METHODS[@]}; do
      ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
      --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_CRT_DIRS} --resume_godin-dir=${GODIN_DIRS[$s]}
    done
  else
    continue
  fi
done

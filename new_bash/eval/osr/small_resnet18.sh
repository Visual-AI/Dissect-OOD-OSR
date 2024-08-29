#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATASETS=("tinyimagenet")
MODEL='resnet18'
OOD_METHODS=("msp" "mls" "energy" "odin" "godin" "sem" "gradnorm" "react_mls" "react_odin" "react_energy")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE" "ARPL_CS" "OE")
CIFAR10_DIRS=("" "" "/disk/work/hjwang/osrd/logs/cifar-10-10_resnet18_OE_conv-default/bestpoint.pth.tar")
CIFAR100_DIRS=("" "" "/disk/work/hjwang/osrd/logs/cifar-10-100_resnet18_OE_conv-default/bestpoint.pth.tar")
TINY_DIRS=("" "" "/disk/work/hjwang/osrd/logs/tinyimagenet_resnet18_OE_closed_rand-augment_9_1_Smoothing0.9/bestpoint.pth.tar")
ARPL_CRT_DIRS=("" "" "" "")
GODIN_DIRS=("/disk/work/hjwang/osrd/logs/cifar-10-10_resnet18_CE_godin_conv-default/bestpoint.pth.tar" "" "/disk/work/hjwang/osrd/logs/cifar-10-100_resnet18_OE_godin_conv-default/bestpoint.pth.tar")
GODIN_ARPL_CRT_DIRS=""

for d in ${!DATASETS[@]}; do
  if [ ${DATASETS[$d]} = 'cifar-10-10' ]; then DIRS=( "${CIFAR10_DIRS[@]}" )
  elif [ ${DATASETS[$d]} = 'tinyimagenet' ]; then DIRS=( "${TINY_DIRS[@]}" )
  else DIRS=( "${CIFAR100_DIRS[@]}" )
  fi
  for s in ${!STRATEGIES[@]}; do
    if [ -f "${DIRS[$s]}" ]
    then
      for i in ${!OOD_METHODS[@]}; do
        ${PYTHON} -m eval.osr_small --model=${MODEL} --in-dataset=${DATASETS[$d]} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --split_idx=0 \
        --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_CRT_DIRS[$d]} \
        --resume_godin-dir=${GODIN_DIRS[$s]} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS} >> ${SAVE_DIR}osr_cifar10_resnet18.out
      done
    else
      continue
    fi
  done
done
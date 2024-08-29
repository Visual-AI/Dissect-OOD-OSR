#!/bin/bash
PYTHON="/home/hjwang/anaconda3/envs/py37/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATASETS=('cifar-10-10' 'cifar-10-100-10' 'cifar-10-100-50' 'tinyimagenet')
MODEL='vit_small'
OOD_METHODS=("mls" "odin" "energy" "react_mls" "react_odin" "react_energy")

STRATEGIES=("CE" "ARPL_CS" "OE")
DIRS=("" "" "")
ARPL_DIRS=("" "" "")

for d in ${!DATASETS[@]}; do
  for s in ${!STRATEGIES[@]}; do
    if [ -f "${DIRS[$s]}" ]
    then
      for i in ${!OOD_METHODS[@]}; do
        ${PYTHON} -m eval.osr_small --model=${MODEL} --in-dataset=${DATASETS[$d]} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --split_idx=0 \
        --resume=${DIRS[$s]} --resume_crit-dir=${ARPL_DIRS[$d]}
      done
    else
      continue
  done
done
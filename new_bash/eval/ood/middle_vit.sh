#!/bin/bash
PYTHON="/home/hjwang/anaconda3/envs/py37/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='middle'
MODEL='vit_small'
OOD_METHODS=("mls" "odin" "energy" "react_mls" "react_odin" "react_energy")

STRATEGIES=("CE" "ARPL_CS" "OE")
DIRS=("" "" "")
ARPL_DIRS=""


for s in ${!STRATEGIES[@]}; do
  if [ -f "${DIRS[$s]}" ]
  then
    for i in ${!OOD_METHODS[@]}; do
      ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
      --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_DIRS}
    done
  else
    continue
  fi
done
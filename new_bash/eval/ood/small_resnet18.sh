#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='small'
MODEL='resnet18'
OOD_METHODS=("msp" "mls" "energy" "odin" "react_mls" "react_odin" "react_energy" "godin" "sem" "gradnorm")
# OOD_METHODS=("odin" "react_odin")
# OOD_METHODS=("ash_mls" "ash_odin" "ash_energy")

MAGs=(0.05)
# MAGs=(1.0 0.5 0.25 0.1 0.05 0.025 0.0125 0.00625 0.003175)

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE")
DIRS=("/disk/work/hjwang/osrd/logs/cifar-10_resnet18_CE_conv-default_Mixup/bestpoint.pth.tar")
ARPL_CRT_DIRS=""

for s in ${!STRATEGIES[@]}; do
  if [ -f "${DIRS[$s]}" ]
  then
    for i in ${!OOD_METHODS[@]}; do
      for m in ${!MAGs[@]}; do
        ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
        --magnitude=${MAGs[$m]}\
        --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_CRT_DIRS} --resume_godin-dir=${GODIN_DIRS[$s]} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS} >> ${SAVE_DIR}ood_cifar10_resnet18.out
      done
    done

  else
    continue
  fi
done

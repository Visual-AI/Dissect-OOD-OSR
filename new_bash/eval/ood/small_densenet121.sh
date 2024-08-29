#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='small'
MODEL='densenet121'
OOD_METHODS=("msp" "mls" "energy" "odin" "react_mls" "react_odin" "react_energy" "godin" "sem" "gradnorm")

MAGs=(0.01)

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE")
DIRS=("/disk/work/hjwang/osrd/logs/cifar-10_densenet121_CE_conv-default/bestpoint.pth.tar")
ARPL_CRT_DIRS=""

GODIN_DIRS=("")
GODIN_ARPL_CRT_DIRS=""

for s in ${!STRATEGIES[@]}; do
  if [ -f "${DIRS[$s]}" ]
  then
    for i in ${!OOD_METHODS[@]}; do
      for m in ${!MAGs[@]}; do
        ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} --val_comb=${DATACOMBS} \
        --magnitude=${MAGs[$m]}\
        --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_CRT_DIRS} --resume_godin-dir=${GODIN_DIRS[$s]} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS} >> ${SAVE_DIR}ood_cifar10_densenet121.out
      done
    done

  else
    continue
  fi
done

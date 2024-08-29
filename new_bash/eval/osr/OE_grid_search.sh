#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATASETS=('scars')
# OOD_METHODS=("msp" "mls" "energy" "odin" "react_mls" "react_odin" "react_energy" "gradnorm")
OOD_METHODS=("mls")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("OE")
CUB_DIRS=("")

SCARS_DIRS=("/disk/work/hjwang/osrd/logs/scars_timm_resnet50_pretrained_OE_OSR_lr=0.005_lamb=0.3/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/scars_timm_resnet50_pretrained_OE_OSR_lr=0.005_lamb=0.5/bestpoint.pth.tar")

AIRCRAFT_DIRS=("")

ARPL_CRT_DIRS=""

for d in ${!DATASETS[@]}; do
  if [ ${DATASETS[$d]} = 'cub' ]; then DIRS=( "${CUB_DIRS[@]}" )
  elif [ ${DATASETS[$d]} = 'scars' ]; then DIRS=( "${SCARS_DIRS[@]}" )
  else DIRS=( "${AIRCRAFT_DIRS[@]}" )
  fi
  for s in ${!STRATEGIES[@]}; do
    for dd in ${!DIRS[@]}; do
      if [ -f "${DIRS[$dd]}" ]
      then
        for i in ${!OOD_METHODS[@]}; do
          echo ${DATASETS[$d]} 
          ${PYTHON} -m eval.osr_large --model='timm_resnet50_pretrained' --in-dataset=${DATASETS[$d]} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} \
          --split_idx=0 --batch_size=32 --feat_dim=2048 --transform='rand-augment' \
          --resume-dir=${DIRS[$dd]} --resume_crit-dir=${ARPL_CRT_DIRS} >> ${SAVE_DIR}oe_tune_SSB_${DATASETS[$d]}.out
        done
      else
        continue
      fi
    done
  done
done
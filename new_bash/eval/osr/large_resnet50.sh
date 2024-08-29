#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATASETS=('cub' 'scars' 'aircraft' 'imagenet21k')
# DATASETS=('waterbird')

OOD_METHODS=("msp" "mls" "energy" "odin" "react_mls" "react_odin" "react_energy" "gradnorm")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("CE" "ARPL" "OE")
CUB_DIRS=("/disk/work/hjwang/osrd/SSB_models/cub/cross_entropy/cub_599_Softmax.pth" "/disk/work/hjwang/osrd/SSB_models/cub/arpl/cub_599_ARPLoss.pth" "")
SCARS_DIRS=("/disk/work/hjwang/osrd/SSB_models/scars/cross_entropy/scars_599_Softmax.pth" "/disk/work/hjwang/osrd/SSB_models/scars/arpl/scars_599_ARPLoss.pth" "")
AIRCRAFT_DIRS=("/disk/work/hjwang/osrd/SSB_models/aircraft/cross_entropy/aircraft_599_Softmax.pth" "/disk/work/hjwang/osrd/SSB_models/aircraft/arpl/aircraft_599_ARPLoss.pth" "")
ARPL_CRT_DIRS=("/disk/work/hjwang/osrd/SSB_models/cub/arpl/cub_599_ARPLoss_criterion.pth" "/disk/work/hjwang/osrd/SSB_models/scars/arpl/scars_599_ARPLoss_criterion.pth" "/disk/work/hjwang/osrd/SSB_models/aircraft/arpl/aircraft_599_ARPLoss_criterion.pth")

for d in ${!DATASETS[@]}; do
  if [ ${DATASETS[$d]} = 'cub' ] || [ ${DATASETS[$d]} = 'waterbird' ]; then DIRS=( "${CUB_DIRS[@]}" )
  elif [ ${DATASETS[$d]} = 'scars' ]; then DIRS=( "${SCARS_DIRS[@]}" )
  else DIRS=( "${AIRCRAFT_DIRS[@]}" )
  fi
  for s in ${!STRATEGIES[@]}; do
    if [ -f "${DIRS[$s]}" ]
    then
      for i in ${!OOD_METHODS[@]}; do
        ${PYTHON} -m eval.osr_large --model='timm_resnet50_pretrained' --in-dataset=${DATASETS[$d]} --ood_method=${OOD_METHODS[$i]} --loss_strategy=${STRATEGIES[$s]} \
        --split_idx=0 --batch_size=32 --feat_dim=2048 --transform='rand-augment' \
        --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_CRT_DIRS[$d]} >> ${SAVE_DIR}osr_SSB_waterbird.out
      done
    else
      continue
    fi
  done
done
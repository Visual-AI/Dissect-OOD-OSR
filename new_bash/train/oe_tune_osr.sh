#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

lr_grid=(0.0005 0.0001)
lamb_grid=(0.1 0.3 0.5 0.7 0.9)

DATASETS=('cub' 'scars' 'aircraft')

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("OE")
CUB_DIRS="/disk/work/hjwang/osrd/SSB_models/cub/cross_entropy/cub_599_Softmax.pth"
SCARS_DIRS="/disk/work/hjwang/osrd/SSB_models/scars/cross_entropy/scars_599_Softmax.pth"
AIRCRAFT_DIRS="/disk/work/hjwang/osrd/SSB_models/aircraft/cross_entropy/aircraft_599_Softmax.pth"


SAVE_DIR=/disk/work/hjwang/osrd/exp/

for d in ${!DATASETS[@]}; do
  if [ ${DATASETS[$d]} = 'cub' ]; then DIRS=${CUB_DIRS}
  elif [ ${DATASETS[$d]} = 'scars' ]; then DIRS=${SCARS_DIRS}
  else DIRS=${AIRCRAFT_DIRS}
  fi
  for l in ${!lr_grid[@]}; do
    for ll in ${!lamb_grid[@]}; do
      echo 'lr='${lr_grid[$l]} ', lamb='${lamb_grid[$ll]}
      ${PYTHON} -m train.oe_osr_tune --model='timm_resnet50_pretrained' --in-dataset=${DATASETS[$d]} --loss_strategy='OE' --lr=${lr_grid[$l]} --lamb=${lamb_grid[$ll]} \
      --split_idx=0 --batch_size=32 --feat_dim=2048 --transform='rand-augment' --resume-dir=${DIRS} >> ${SAVE_DIR}oe_tune_YFCC_${DATASETS[$d]}.out
    done
  done
done
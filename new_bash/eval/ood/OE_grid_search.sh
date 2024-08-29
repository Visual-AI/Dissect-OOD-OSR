#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATACOMBS='large'
MODEL='resnet50'
# OOD_METHODS=("msp" "mls" "energy" "odin" "react_mls" "react_odin" "react_energy" "gradnorm")
OOD_METHODS=("mls")

SAVE_DIR=/disk/work/hjwang/osrd/exp/

STRATEGIES=("OE")
DIRS=("/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.001_lamb=0.1/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.001_lamb=0.3/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.001_lamb=0.5/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.001_lamb=0.7/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.001_lamb=0.9/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.0001_lamb=0.1/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.0001_lamb=0.3/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.0001_lamb=0.5/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.0001_lamb=0.7/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.0001_lamb=0.9/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.00001_lamb=0.1/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.00001_lamb=0.3/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.00001_lamb=0.5/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.00001_lamb=0.7/bestpoint.pth.tar" \
"/disk/work/hjwang/osrd/logs/imagenet_resnet50_OE_None_lr=0.00001_lamb=0.9/bestpoint.pth.tar")
ARPL_DIRS=""

GODIN_DIRS=("" "")
GODIN_ARPL_CRT_DIRS=""

for s in ${!DIRS[@]}; do
  for i in ${!OOD_METHODS[@]}; do
    ${PYTHON} -m eval.ood --model=${MODEL} --ood_method=${OOD_METHODS[$i]} --loss_strategy='OE' --val_comb=${DATACOMBS} \
    --resume-dir=${DIRS[$s]} --resume_crit-dir=${ARPL_DIRS} --resume_godin-dir=${GODIN_DIRS[$s]} --resume_godin_crit-dir=${GODIN_ARPL_CRT_DIRS} >> ${SAVE_DIR}ood_SSB_OE.out
  done
done
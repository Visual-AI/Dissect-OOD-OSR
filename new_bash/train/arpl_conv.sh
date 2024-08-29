#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

# DATASETS=("cifar-10-10" "cifar-10-100-10" "tinyimagenet")
# DATASETS=("cifar-10" "cifar-100")
DATASETS=("cifar-10")


MODEL='resnet18'
# MODEL='resnet50'

CONFIG=("conv-default")
# CONFIG=("conv-default" "conv-cifar10-default")
# CONFIG=("closed" "closed")
# CONFIG=("godin_conv-default" "godin_conv-cifar10-default")

for d in ${!DATASETS[@]}; do
  ${PYTHON} -m train.arpl_conv --in-dataset=${DATASETS[$d]} --loss_strategy='ARPL_CS' --model=${MODEL} --ablation=${CONFIG[$d]}
done
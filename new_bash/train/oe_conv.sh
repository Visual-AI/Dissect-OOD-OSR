#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

# MODEL='resnet18'
MODEL='resnet50'
# MODEL='wrn'
# MODEL='densenet121'

# DATASETS=("cifar-10-10" "cifar-10-100-10" "tinyimagenet")
# CONFIG=("conv-default" "conv-default" "closed")
# CONFIG=("godin_conv-default" "godin_conv-default" "godin_closed")

# DATASETS=("cifar-10" "cifar-100")
DATASETS=("imagenet")
# CONFIG=("conv-default" "conv-default")
# CONFIG=("closed" "closed")
CONFIG=("oe")
# CONFIG=("oe-default" "oe-default")
# CONFIG=("godin_wrn-oe" "godin_wrn-oe")
# CONFIG=("godin_conv-default" "godin_conv-default")

for d in ${!DATASETS[@]}; do
  # ${PYTHON} -m train.oe_conv --in-dataset=${DATASETS[$d]} --loss_strategy='OE' --model=${MODEL} --ablation=${CONFIG[$d]}
  ${PYTHON} -m train.oe_conv_scratch --in-dataset=${DATASETS[$d]} --loss_strategy='OE' --model=${MODEL} --ablation=${CONFIG[$d]}
done
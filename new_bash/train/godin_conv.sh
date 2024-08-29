#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

# DATASETS='cifar-10-10 cifar-10-100-10 cifar-10-100-50 tinyimagenet'
# DATASETS='cifar-10 cifar-100'
DATASETS='cifar-10'

for dataset in $DATASETS; do
  ${PYTHON} -m train.godin_conv --in-dataset=${dataset} --loss_strategy='CE' --model='resnet18'
done
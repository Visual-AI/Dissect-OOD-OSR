#!/bin/bash
PYTHON="/home/hjwang/anaconda3/envs/py37/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

# DATASETS='cifar-10-10 cifar-10-100-10 cifar-10-100-50 tinyimagenet'
DATASETS='cifar-10 cifar-100'

for dataset in $DATASETS; do
  ${PYTHON} -m train.ce_vit --in-dataset=${dataset} --loss_strategy='CE' --model='vit_small'
done
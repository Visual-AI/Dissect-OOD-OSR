#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/osr/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

# lr_grid=(0.0005 0.0001)
# lr_grid=(0.0005)
lr_grid=(0.0001)

lamb_grid=(0.1 0.3 0.5 0.7 0.9)

SAVE_DIR=/disk/work/hjwang/osrd/exp/

for l in ${!lr_grid[@]}; do
  for ll in ${!lamb_grid[@]}; do
    echo 'lr='${lr_grid[$l]} ', lamb='${lamb_grid[$ll]}
    ${PYTHON} -m train.oe_conv_tune --loss_strategy='OE' --lr=${lr_grid[$l]} --lamb=${lamb_grid[$ll]} --ablation='ImagenetR' >> ${SAVE_DIR}oe_tune_imagenetR.out
  done
done
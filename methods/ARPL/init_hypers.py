import os
import torch
import random
import numpy as np
import inspect

import os

def get_default_hyperparameters(args):
    """
    Adjusts args to match parameters used in paper: https://arxiv.org/abs/2110.06207
    """

    # DATASET / LOSS specific hyperparams (Image Size   Learning Rate   RandAug N   RandAug M   Label Smoothing Batch Size)
    if args.in_dataset == 'mnist':
        args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 32, 0.1, 1, 8, 0, 128

    elif args.in_dataset == 'svhn':
        args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 32, 0.1, 1, 18, 0, 128

    elif args.in_dataset == 'cifar-10':
        args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 32, 0.1, 1, 6, 0, 128

    elif args.in_dataset == 'cifar-10-10':
        if args.loss == 'Softmax':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 32, 0.1, 1, 6, 0, 128
        elif args.loss == 'ARPLoss':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 32, 0.1, 1, 15, 0, 128

    elif args.in_dataset == 'cifar-10-100':
        args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 32, 0.1, 1, 6, 0, 128

    elif args.in_dataset == 'tinyimagenet':
        if args.loss == 'Softmax':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 64, 0.01, 1, 9, 0.9, 128
        elif args.loss == 'ARPLoss':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 64, 0.001, 1, 9, 0.9, 128

    elif args.in_dataset == 'cub':
        if args.loss == 'Softmax':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 448, 0.001, 2, 30, 0.3, 32
        elif args.loss == 'ARPLoss':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 448, 0.001, 2, 30, 0.2, 32

    elif args.in_dataset == 'aircraft':
        if args.loss == 'Softmax':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 448, 0.001, 2, 15, 0.2, 32
        elif args.loss == 'ARPLoss':
            args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = 448, 0.001, 2, 18, 0.1, 32


    # if args.in_dataset in ('cub', 'aircraft', 'scars', 'imagenet'):
    #     args.model = 'timm_resnet50_pretrained'
    #     args.resnet50_pretrain = 'places_moco'
    #     args.feat_dim = 2048

    # else:
    #     args.model = 'classifier32'
    #     args.feat_dim = 128

    return args
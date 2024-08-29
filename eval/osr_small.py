import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from methods.ARPL.arpl_models.resnetABN import resnet18ABN
from methods import test

from utils.utils import seed_torch, str2bool, strip_state_dict

# from methods.ARPL.init_hypers import get_default_hyperparameters
from data.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model
from utils.stools import get_GMM_stat

# from config import exp_root

parser = argparse.ArgumentParser("OSR_eval")

# Dataset
parser.add_argument('--in-dataset', type=str, default='cub', help="")
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=32)

# optimization
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str)
parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

# misc
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--resume-dir',type=str, default='', help='dir of output')
parser.add_argument('--resume_godin-dir',type=str, default='', help='dir of output')
parser.add_argument('--resume_crit-dir',type=str, default='', help='dir of output')
parser.add_argument('--resume_godin_crit-dir',type=str, default='', help='dir of output')

parser.add_argument('--ood_method', default='mls', type=str, help="mls/odin/energy/react")
parser.add_argument('--loss_strategy', default='', type=str, help="CE/OE/ARPL/ARPL_CS")

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.image_size = 64 if args.in_dataset == 'tinyimagenet' else 32

if 'cifar-10-100' in args.in_dataset:
    args.out_num = int(args.in_dataset.split('-')[-1])
    args.in_dataset = 'cifar-10-100'


def get_model(num_classes):
    # MODEL
    if args.model == 'resnet18':
        if args.in_dataset == 'tinyimagenet':
            from methods.ARPL.arpl_models.resnet import ResNet18
            model = ResNet18(num_c=num_classes, feat_dim=args.feat_dim) 
        else:
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, feat_dim=args.feat_dim)
    elif args.model == 'vit_small':
        from models.vit_small import ViT
        model = ViT(image_size=args.image_size, patch_size=4, num_classes=len(args.train_classes), dim=512, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model)

    if args.ood_method == 'godin':
        from models.godin_net import UpperNet
        model = UpperNet(model, args.feat_dim, num_classes)

    return model


def main():
    if args.ood_method == 'godin':
        print(args.resume_godin_dir)
    else:
        print(args.resume_dir)

    results = dict()

    # SEED
    seed_torch(args.seed)

    # DATASETS
    args.train_classes, args.open_set_classes = get_class_splits(args.in_dataset, args.split_idx, cifar_plus_n=args.out_num)
    datasets = get_datasets(args.in_dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                            split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                            args=args)

    # DATALOADER
    dataloaders = {}
    for k, v, in datasets.items():
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=16)

    # SAVE PARAMS
    options = vars(args)
    options.update(
        {
            'known':    args.train_classes,
            'unknown':  args.open_set_classes,
            'img_size': args.image_size,
            'dataloaders': dataloaders,
            'num_classes': len(args.train_classes)
        }
    )

    # DATALOADERS
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    if 'CS' in args.loss_strategy:
        net = resnet18ABN(num_classes=options['num_classes'], num_bns=2)
    else:
        net = get_model(options['num_classes'])

    # GET LOSS
    if 'ARPL' in args.loss_strategy:
        args.loss = "ARPLoss"
    Loss = importlib.import_module('methods.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)
    
    # PREPARE EXPERIMENT
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if args.model == 'vit_small':
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(args.resume_dir)['model'])
    else:
        if 'ARPL' in args.loss_strategy:
            net, criterion = load_networks(net, options=options, model_dir=args.resume_dir, crit_dir=args.resume_crit_dir, criterion=criterion)
        elif args.ood_method == 'godin':
            if os.path.isfile(args.resume_godin_dir):
                state_dict = strip_state_dict(torch.load(args.resume_godin_dir))
                net.load_state_dict(state_dict)
            else:
                assert False, 'Not exist file.'
        else:
            state_dict = strip_state_dict(torch.load(args.resume_dir))
            net.load_state_dict(state_dict)

    if args.ood_method == 'sem':
        feature_type_list = ['stat', 'mean', 'mean', 'mean', 'flat']
        reduce_dim_list = ['pca_50', 'none', 'none', 'none', 'pca_50']
        num_clusters_list = [3, 1, 1, 1, 10]
        feature_mean, feature_prec, component_weight_list, transform_matrix_list = get_GMM_stat(net, trainloader, num_clusters_list, feature_type_list, reduce_dim_list, options)
        options.update(
            {
                'feature_type_list': feature_type_list,
                'feature_mean': feature_mean,
                'feature_prec': feature_prec,
                'component_weight_list': component_weight_list,
                'transform_matrix_list': transform_matrix_list
            }
        )

    res = test(net, criterion, testloader, outloader, **options)

    # LOG
    res['Loss'] = args.loss_strategy
    res['Ood_method'] = args.ood_method
    res['Dataset'] = args.in_dataset + '-' + str(args.out_num) if 'cifar' in args.in_dataset else args.in_dataset
    
    # res['split_idx'] = args.split_idx
    # res['unknown'] = args.open_set_classes
    # res['known'] = args.train_classes
    # res['ID'] = args.log_dir.split("/")[-1]
    print(res)



if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import importlib
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.open_set_datasets import get_class_splits, get_datasets
from methods.OOD.util.data_loader import get_loader_in, get_loader_out
from utils.tinyimages_80mn_loader import TinyImages
from utils.yfcc_ImageFolder import ImageFolder

from methods import train_oe, test_id
from utils.utils import seed_torch, str2bool

from utils.schedulers import get_scheduler, WarmUpLR

import sys
from os import path

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier with OE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir',type=str, default='/disk/datasets/ood_zoo',help='dir of output')
parser.add_argument('--yfcc-dir',type=str, default='/disk/datasets',help='dir of output')
parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')
parser.add_argument('--config', nargs="?", type=str, default="/disk/work/hjwang/osrd/train_configs.yaml", help="Configuration file to use")

parser.add_argument('--in-dataset', type=str, default='cifar10', help='Choose between cifar-10-10, cifar-10-100-10, cifar-10-100-50, tinyimagenet')
parser.add_argument('--model', type=str, default='resnet18', help='Choose architecture.')
parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim, only for classifier32 at the moment")

# Optimization options
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, help='The initial learning rate.')
parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 penalty).')
parser.add_argument('--batch_size', '-b', type=int, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--scheduler', type=str)
parser.add_argument('--lamb', type=float, default=0.5, help='The balance factor.')

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")

parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--temp', type=float, default=1.0, help="temp")

parser.add_argument('--using_yfcc', default=False, type=str2bool, help='Do we use YFCC data', metavar='BOOL')
parser.add_argument('--loss', type=str, default='OELoss')
parser.add_argument('--loss_strategy', default='OE', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--ablation', type=str, help='conv-default/oe-default')

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.image_size = 64 if args.in_dataset == 'tinyimagenet' else 32
mean = [0.4914, 0.4822, 0.4465] if not args.in_dataset == 'tinyimagenet' else [0.485, 0.456, 0.406]
std = [0.2023, 0.1994, 0.2010] if not args.in_dataset == 'tinyimagenet' else [0.229, 0.224, 0.225]


if args.in_dataset in ['cifar-10', 'cifar-100']:
    args.is_ood = True
else:
    args.is_ood = False

if 'cifar-10-100' in args.in_dataset:
    args.out_num = int(args.in_dataset.split('-')[-1])
    args.in_dataset = 'cifar-10-100'

with open(args.config) as fp:
    ds_list = args.in_dataset.split('-')
    if len(ds_list) < 2:
        ds = ds_list[0]
    else:
        ds = ds_list[0] + '-' + ds_list[1]
    if 'conv-default' in args.ablation:
        cfg_name = 'conv-default'
    elif 'closed' in args.ablation:
        cfg_name = 'closed'    
    elif 'oe-default' in args.ablation:
        cfg_name = 'oe-default'    
    cfg = yaml.safe_load(fp)[ds][cfg_name]

args.epochs = cfg.get("epochs")
args.batch_size = cfg.get("batch_size")
args.scheduler = cfg.get("scheduler")
args.optim = cfg.get("optim")
args.lr, args.weight_decay = cfg.get("lr"), cfg.get("weight_decay")
args.is_nesterov = cfg.get("is_nesterov")
args.steps, args.gamma = cfg.get("steps"), cfg.get("gamma")
args.rand_aug_m, args.rand_aug_n = cfg.get("rand_aug_m"), cfg.get("rand_aug_n")
args.label_smoothing = cfg.get("label_smoothing")


def get_model(num_classes):
    # Create model
    if args.model == 'resnet18':
        if args.in_dataset == 'tinyimagenet':
            from methods.ARPL.arpl_models.resnet import ResNet18
            model = ResNet18(num_c=num_classes, feat_dim=args.feat_dim) 
        else:
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, feat_dim=args.feat_dim)
    elif args.model == 'resnet50':
        from models.resnet import resnet50_cifar
        model = resnet50_cifar(num_classes=num_classes, feat_dim=args.feat_dim)
    elif args.model == 'wrn':
        from models.wrn import WideResNet
        model = WideResNet(40, num_classes, 2, dropRate=0.3)
        args.feat_dim = 64 * 2
    elif args.model == 'densenet121':
        from models.densenet import densenet121
        model = densenet121(num_classes=num_classes)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model)
    
    model = model.to(args.device)
    if 'godin' in args.ablation:
        from models.godin_net import UpperNet
        model = UpperNet(model, args.feat_dim, num_classes)
    return model


def get_optimizer(params_list, weight_decay=None):
    if args.in_dataset == 'tinyimagenet':
        optimizer = torch.optim.Adam(params_list, lr=args.lr) 
    else:
        if weight_decay is None:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, nesterov=args.is_nesterov)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=weight_decay, nesterov=args.is_nesterov)
    return optimizer


if __name__ == '__main__':
    if args.is_ood:
        loader_in_dict = get_loader_in(args)
        trainloader, testloader, args.num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes 
        options = vars(args)
        options.update(
            {
                'img_size': args.image_size,
                'dataloaders': loader_in_dict,
                'train_len': len(trainloader),
                'num_classes': args.num_classes
            }
        )
    else:
        # DATASETS
        args.train_classes, args.open_set_classes = get_class_splits(args.in_dataset, args.split_idx, cifar_plus_n=args.out_num)
        datasets = get_datasets(args.in_dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)

        # RANDAUG HYPERPARAM SWEEP
        if args.transform == 'rand-augment':
            if args.rand_aug_m is not None:
                if args.rand_aug_n is not None:
                    datasets['train'].transform.transforms[0].m = args.rand_aug_m
                    datasets['train'].transform.transforms[0].n = args.rand_aug_n

        # DATALOADER
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            batch_size = args.batch_size
            dataloaders[k] = DataLoader(v, batch_size=batch_size, shuffle=shuffle, sampler=None, num_workers=16)

        # DATALOADERS
        trainloader = dataloaders['train']
        testloader = dataloaders['val']
        outloader = dataloaders['test_unknown']

        # SAVE PARAMS
        options = vars(args)
        options.update(
            {
                'known':    args.train_classes,
                'unknown':  args.open_set_classes,
                'img_size': args.image_size,
                'train_len': len(trainloader),
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )

    args.ablation = '{}_{}_{}_{}'.format(args.in_dataset, args.model, args.loss_strategy, args.ablation)
    if args.transform == 'rand-augment' and args.rand_aug_m is not None and args.rand_aug_n is not None:
        args.ablation = '{}_{}_{}_{}'.format(args.ablation, args.transform, args.rand_aug_m, args.rand_aug_n)
    if args.label_smoothing is not None:
        args.ablation = '{}_Smoothing{}'.format(args.ablation, args.label_smoothing)
    if args.using_yfcc:
        args.ablation = '{}_yfcc'.format(args.ablation)
    args.out_dir = os.path.join(args.out_dir, args.ablation)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print(args.out_dir)
    # cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

    if args.using_yfcc:
        oeloader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.yfcc_dir, 'clip_data'), transform=trn.Compose(
                [trn.Resize(args.image_size), trn.CenterCrop(args.image_size), trn.ToTensor(), trn.Normalize(mean, std)])), 
            batch_size=args.oe_batch_size, shuffle=True)
    else:
        if args.in_dataset == 'tinyimagenet':
            ood_data = TinyImages(os.path.join(args.data_dir, 'oe_data', '300K_random_images.npy'), transform=trn.Compose(
                [trn.ToTensor(), trn.ToPILImage(), trn.Resize(64), trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)]))        
        else:
            ood_data = TinyImages(os.path.join(args.data_dir, 'oe_data', '300K_random_images.npy'), transform=trn.Compose(
                [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
                trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))

        oeloader = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # GET LOSS
    criterion = getattr(importlib.import_module('methods.loss.'+'Softmax'), 'Softmax')(**options)

    # Get base network
    net = get_model(num_classes=options['num_classes'])

    # GET SCHEDULER
    parameters = []
    parameters_h = []
    for name, parameter in net.named_parameters():
        if 'dist' in name and 'godin' in args.ablation:
            parameters_h.append(parameter)
        else:
            parameters.append(parameter)

    optimizer = get_optimizer(params_list=[{'params': parameters}, {'params': criterion.parameters()}], weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args, options)
    optimizer_h = None
    if 'godin' in args.ablation:
        optimizer_h = get_optimizer(params_list=[{'params': parameters_h}, {'params': criterion.parameters()}])
        scheduler_h = get_scheduler(optimizer_h, args, options)

    warmup_scheduler = WarmUpLR(optimizer, len(trainloader))

    best_id_acc = 0 

    # Main loop
    for epoch in range(0, args.epochs):
        train_oe(net, optimizer, scheduler, warmup_scheduler, trainloader, oeloader, epoch, optimizer_h=optimizer_h, **options)
        id_acc = test_id(net, criterion, testloader, **options)
        print(args.out_dir)
        print("Epoch {}: Acc (%): {:.3f}\t".format(epoch, id_acc))
        
        if best_id_acc < id_acc:
            torch.save(net.state_dict(), os.path.join(args.out_dir, 'bestpoint.pth.tar'))
            best_id_acc = id_acc
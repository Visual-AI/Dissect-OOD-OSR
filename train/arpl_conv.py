import os
import argparse
import datetime
import time
import pandas as pd
import importlib
import yaml

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from methods.ARPL.arpl_models import gan
from methods import train, train_cs, test_id

from utils.utils import seed_torch, str2bool, save_networks

from methods.ARPL.init_hypers import get_default_hyperparameters
from utils.schedulers import get_scheduler, WarmUpLR

from data.open_set_datasets import get_class_splits, get_datasets
from methods.OOD.util.data_loader import get_loader_in, get_loader_out


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--data-dir',type=str, default='/disk/datasets/ood_zoo',help='dir of output')
parser.add_argument('--config', nargs="?", type=str, default="/disk/work/hjwang/osrd/train_configs.yaml", help="Configuration file to use")
parser.add_argument('--in-dataset', type=str, default='cub', help="")
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=32)

# optimization
parser.add_argument('--optim', type=str, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int)

parser.add_argument('--lr', type=float, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--epochs', type=int)
parser.add_argument('--scheduler', type=str)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='ARPLoss')
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
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')
parser.add_argument('--ood_method', default='mls', type=str, help="mls/odin/energy/react")
parser.add_argument('--loss_strategy', default='ARPL_CS', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--ablation', type=str, help='conv-default/oe-default')

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.image_size = 64 if args.in_dataset == 'tinyimagenet' else 32

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
    elif 'conv-cifar10-default' in args.ablation:
        cfg_name = 'conv-cifar10-default'
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
    if args.model == 'resnet18':
        from methods.ARPL.arpl_models.resnetABN import resnet18ABN
        model = resnet18ABN(num_classes=num_classes, feat_dim=args.feat_dim)
    elif args.model == 'resnet50':
        from methods.ARPL.arpl_models.resnetABN import resnet18ABN
        model = resnet50ABN(num_classes=num_classes, feat_dim=args.feat_dim)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model)
    model = model.to(args.device)
    if 'godin' in args.ablation:
        from models.godin_net import UpperNet
        model = UpperNet(model, args.feat_dim, num_classes)
    return model


def get_optimizer(params_list, weight_decay=None):
    if args.optim == 'adam' or args.in_dataset == 'tinyimagenet':
        optimizer = torch.optim.Adam(params_list, lr=args.lr) 
    else:
        if weight_decay is None:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, nesterov=args.is_nesterov)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=weight_decay, nesterov=args.is_nesterov)
    return optimizer



if __name__ == '__main__':
    # DATASETS
    if args.is_ood:
        loader_in_dict = get_loader_in(args)
        trainloader, testloader, args.num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes 
        options = vars(args)
        options.update(
            {
                'img_size': args.image_size,
                'dataloaders': loader_in_dict,
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
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )
        
    # TRAIN
    args.ablation = '{}_{}_{}_{}'.format(args.in_dataset, args.model, args.loss_strategy, args.ablation)
    if args.transform == 'rand-augment' and args.rand_aug_m is not None and args.rand_aug_n is not None:
        args.ablation = '{}_{}_{}_{}'.format(args.ablation, args.transform, args.rand_aug_m, args.rand_aug_n)
    if args.label_smoothing is not None:
        args.ablation = '{}_Smoothing{}'.format(args.ablation, args.label_smoothing)
    args.out_dir = os.path.join(args.out_dir, args.ablation)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print(args.out_dir)
    # cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

    # -----------------------------
    # --CS MODEL AND LOSS
    # -----------------------------
    print("Creating GAN")
    nz, ns = args.nz, 1
    if args.image_size >= 64:
        netG = gan.Generator(1, nz, 64, 3)
        netD = gan.Discriminator(1, 3, 64)
    else:
        netG = gan.Generator32(1, nz, 64, 3)
        netD = gan.Discriminator32(1, 3, 64)
    netG = netG.to(args.device)
    netD = netD.to(args.device)
    fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
    criterionD = nn.BCELoss()

    # GET LOSS
    Loss = importlib.import_module('methods.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)
    criterion = criterion.to(args.device)
    
    # Get base network
    net = get_model(num_classes=options['num_classes'])

    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    fixed_noise.to(args.device)

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

    # TRAIN
    for epoch in range(args.epochs):
        train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader, optimizer_h=optimizer_h, **options)
        train(net, criterion, optimizer, warmup_scheduler, trainloader, **options)
        id_acc = test_id(net, criterion, testloader, **options)
        print(args.out_dir)
        print("Epoch {}: Acc (%): {:.3f}\t".format(epoch, id_acc))

        if best_id_acc < id_acc:
            save_networks(net, args.out_dir, 'bestpoint.pth.tar', options, criterion=criterion)
            best_id_acc = id_acc

        # STEP SCHEDULER
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(id_acc, epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)
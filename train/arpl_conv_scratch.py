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

from methods.ARPL.arpl_models import gan
from methods.ARPL.arpl_models.resnetABN import resnet18ABN
from methods.ARPL.arpl_utils import save_networks
from methods.ARPL.core import train, train_cs, test

from utils.utils import seed_torch, str2bool

from methods.ARPL.init_hypers import get_default_hyperparameters
from utils.schedulers import get_scheduler
from models.model_utils import get_model

from data.open_set_datasets import get_class_splits, get_datasets
from methods.OOD.util.data_loader import get_loader_in, get_loader_out

from config import exp_root

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--in-dataset', type=str, default='cub', help="")
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=32)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')

parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str)
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco', help='Which pretraining to use if --model=timm_resnet50_pretrained. Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=None, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

# misc
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--train_feat_extractor', default=True, type=str2bool, help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--resume', type=str2bool, default=False, help='whether to resume training')

parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')
# parser.add_argument('--resume-dir',type=str, default=None, help='dir of output')

parser.add_argument('--ood_method', default='mls', type=str, help="mls/odin/energy/react")
parser.add_argument('--loss_strategy', default='ARPL_CS', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--config_strategy', default='bas_config', type=str, help="bas_config/mls_config")
parser.add_argument('--is_pretrained', default=False, type=str2bool, help='Do we use ImageNet pretrained model', metavar='BOOL')
parser.add_argument('--ablation', type=str, default='.', help='ablation study')

args = parser.parse_args()
args.image_size = 64 if args.in_dataset == 'tinyimagenet' else 32

if args.in_dataset in ['cifar-10', 'cifar-100']:
    args.is_ood = True
else:
    args.is_ood = False

if 'cifar-10-100' in args.in_dataset:
    args.out_num = int(args.in_dataset.split('-')[-1])
    args.in_dataset = 'cifar-10-100'

if args.config_strategy == 'bas_config':
    args.epochs = 100
    args.scheduler = 'multi_step'
    args.steps = [30, 60, 90, 120]
    args.batch_size, args.oe_batch_size = 128, 256
    args.optim = 'sgd'

    if args.in_dataset == 'tinyimagenet':
        args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0.9, 1, 9
        args.optim = 'adam'
        args.lr = 0.001
    else:
        args.lr = 0.1

elif args.config_strategy == 'mls_config':
    args.epochs = 600
    args.scheduler = 'cosine_warm_restarts_warmup'
    args.optim = 'sgd'
    args.batch_size, args.oe_batch_size = 128, 256
    args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0, 1, 6
    
    if args.in_dataset == 'tinyimagenet':
        args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0.9, 1, 9
        args.optim = 'adam'
        args.lr = 0.001  
    elif args.in_dataset == 'cifar-10-100':
        args.lr = 0.1
    else:
        args.lr = 0.1


def get_optimizer(args, params_list):
    if args.optim is None:
        optimizer = torch.optim.Adam(params_list, lr=args.lr) if args.in_dataset == 'tinyimagenet' else torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=args.lr)
    return optimizer



if __name__ == '__main__':
    # args = get_default_hyperparameters(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = dict()

    exp_root = os.path.join(exp_root, args.ablation)
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    args.log_dir = os.path.join(exp_root, 'logs')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # INIT
    args.feat_dim = 512    

    # DATASETS
    if args.is_ood:
        loader_in_dict = get_loader_in(args)
        trainloader, testloader, args.num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes 
    else:
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
            batch_size = 1 if not k == 'train' and args.ood_method =='gradnorm' else args.batch_size
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

    dir_path = os.path.join('/'.join(args.log_dir.split("/")[:-2]), 'results')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if args.in_dataset == 'cifar-10-100':
        file_name = '{}_{}_{}_{}_cs.csv'.format(args.in_dataset, args.out_num, args.model, args.loss)
    else:
        file_name = '{}_{}_{}_cs.csv'.format(args.in_dataset, args.model, args.loss)

    # TRAIN
    if args.transform == 'rand-augment' and args.rand_aug_m is not None and args.rand_aug_n is not None:
        args.ablation = '{}_{}_{}_{}'.format(args.transform, args.rand_aug_m, args.rand_aug_n, args.ablation)
    if args.label_smoothing is not None:
        args.ablation = 'Smoothing{}_{}'.format(args.label_smoothing, args.ablation)
    args.ablation = '{}_{}_{}_{}_{}'.format(args.in_dataset, args.model, args.loss_strategy, args.config_strategy, args.ablation)
    args.out_dir = os.path.join(args.out_dir, args.ablation)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print(args.out_dir)
    # cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

    oeloader = None

    # MODEL
    print("Creating model: {}".format(args.model))
    net = resnet18ABN(num_classes=len(args.train_classes))
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
    Loss = importlib.import_module('methods.ARPL.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)
    
    # PREPARE EXPERIMENT
    net = net.to(args.device)
    # net = nn.DataParallel(net).to(args.device)
    criterion = criterion.to(args.device)
    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=[{'params': net.parameters()}, {'params': criterion.parameters()}])

    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    fixed_noise.to(args.device)

    # GET SCHEDULER
    scheduler = get_scheduler(optimizer, args)

    best_AUROC, best_avg_ACC, best_results = 0, 0, 0

    # TRAIN
    for epoch in range(args.epochs):
        train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader, epoch=epoch, **options)
        train(net, criterion, optimizer, trainloader, oeloader, epoch=epoch, **options)

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.epochs:
            results = test(net, criterion, trainloader, testloader, outloader, epoch=epoch, **options)
            print(args.out_dir)
            print("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t".format(epoch, results['ACC'], results['AUROC']))
            if best_avg_ACC < results['ACC']:
                torch.save(net.state_dict(), os.path.join(args.out_dir, 'bestpoint.pth.tar'))
                results['Info'] = "The best model at {}-th epoch".format(epoch)
                best_avg_ACC = results['ACC']
                best_results = results

        # STEP SCHEDULER
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results['ACC'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)
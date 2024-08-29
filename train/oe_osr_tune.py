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

from methods.ARPL.arpl_models.wrapper_classes import TimmResNetWrapper

from methods import train_osr_oe
from utils.utils import seed_torch, str2bool, strip_state_dict, load_networks
from models.model_utils import get_model
from config import osr_split_dir

from utils.schedulers import get_scheduler, WarmUpLR
import timm
import sys
from os import path

parser = argparse.ArgumentParser(description='Finerue a ImageNet Classifier with OE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir',type=str, default='/disk/datasets/ood_zoo',help='dir of output')
parser.add_argument('--yfcc-dir',type=str, default='/disk/datasets',help='dir of output')
parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')
parser.add_argument('--config', nargs="?", type=str, default="/home/hjwang/osrd/train_configs.yaml", help="Configuration file to use")

parser.add_argument('--in-dataset', type=str, default='imagenet', help='Choose between cifar-10-10, cifar-10-100-10, cifar-10-100-50, tinyimagenet')
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--model', type=str, default='resnet50', help='Choose architecture.')
parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim, only for classifier32 at the moment")

# Optimization options
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, help='The initial learning rate.')
parser.add_argument('--lamb', type=float, help='The balance factor.')

parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 penalty).')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--scheduler', type=str)

parser.add_argument('--train_feat_extractor', default=True, type=str2bool, help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--is_pretrained', default=False, type=str2bool, help='Do we use ImageNet pretrained model', metavar='BOOL')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco', help='Which pretraining to use if --model=timm_resnet50_pretrained. Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')

# misc
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--resume-dir',type=str, default='', help='dir of output')

parser.add_argument('--loss', type=str, default='OELoss')
parser.add_argument('--loss_strategy', default='OE', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--ablation', type=str, help='conv-default/oe-default')

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

args.epochs = 20
args.scheduler = "cosine_warm_restarts_warmup"
args.optim = "sgd"
args.weight_decay = 5.0e-4
args.is_nesterov = True
args.steps, args.gamma = None, None
args.rand_aug_m, args.rand_aug_n = None, None
args.label_smoothing = None

args.ablation = 'OSR_yfcc'

def test_id(net, inloader, **options):
    net.eval()
    correct, total = 0, 0
    torch.cuda.empty_cache()

    id_pbar = tqdm(inloader)
    
    with torch.no_grad():
        for batch_idx, tuples in enumerate(id_pbar):
            if len(tuples) == 2:
                data, labels = tuples
            elif len(tuples) == 3:
                data, labels, idx = tuples
            data, labels = data.to(options['device']), labels.to(options['device'])
            
            with torch.no_grad():
                logits, _ = net(data, True)
                total += labels.size(0)
                correct += (logits.data.max(1)[1] == labels.data).sum()

        id_acc = float(correct) * 100. / float(total)

    return id_acc


def get_optimizer(params_list, weight_decay=None):
    if weight_decay is None:
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, nesterov=args.is_nesterov)
    else:
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=weight_decay, nesterov=args.is_nesterov)
    return optimizer


if __name__ == '__main__':
    options = vars(args)

    # Get OSR splits
    osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.in_dataset))

    with open(osr_path, 'rb') as f:
        class_info = pickle.load(f)

    train_classes = class_info['known_classes']

    args.train_classes = train_classes

    datasets = get_datasets(args.in_dataset, transform=args.transform, train_classes=args.train_classes,
                            image_size=args.image_size, balance_open_set_eval=False,
                            split_train_val=False)

    dataloaders = {}
    for k, v, in datasets.items():
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=8)

    options.update(
        {
            'known':    args.train_classes,
            'img_size': args.image_size,
            'dataloaders': dataloaders,
            'num_classes': len(args.train_classes)
        }
    )

    # GET LOSS
    if 'ARPL' in args.loss_strategy:
        args.loss = "ARPLoss"
    Loss = importlib.import_module('methods.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)    
    criterion = criterion.to(args.device)

    # DATALOADERS
    trainloader = dataloaders['train']
    testloader = dataloaders['test_known']
    outloader = dataloaders['test_unknown']

    args.ablation = '{}_{}_{}_{}'.format(args.in_dataset, args.model, args.loss_strategy, args.ablation)
    args.ablation = '{}_lr={}_lamb={}'.format(args.ablation, args.lr, args.lamb)
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

    transform_train_largescale = trn.Compose([
        trn.Resize(256),
        trn.RandomResizedCrop(224),
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std)])

    # oeloader = torch.utils.data.DataLoader(
    #             dset.ImageFolder(os.path.join(args.data_dir, 'oe_data', 'Places'), transform=transform_train_largescale), 
    #             batch_size=args.oe_batch_size, shuffle=True)

    oeloader = torch.utils.data.DataLoader(
                dset.ImageFolder(os.path.join(args.yfcc_dir, 'clip_data'), transform=transform_train_largescale), 
                batch_size=args.oe_batch_size, shuffle=True)
    # GET LOSS
    criterion = getattr(importlib.import_module('methods.loss.'+'Softmax'), 'Softmax')(**options)

    # Get base network
    net = get_model(args, wrapper_class=None, evaluate=True)
    filename = strip_state_dict(torch.load(args.resume_dir))
    net.load_state_dict(filename)
    net = net.to(args.device)

    # GET SCHEDULER
    parameters = []
    parameters_h = []
    for name, parameter in net.named_parameters():
        parameters.append(parameter)

    optimizer = get_optimizer(params_list=[{'params': parameters}, {'params': criterion.parameters()}], weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args, options)
    optimizer_h = None

    warmup_scheduler = WarmUpLR(optimizer, len(trainloader))

    best_id_acc = 0 

    # Main loop
    for epoch in range(0, args.epochs):
        train_osr_oe(net, optimizer, scheduler, warmup_scheduler, trainloader, oeloader, epoch, optimizer_h=optimizer_h, **options)
        id_acc = test_id(net, testloader, **options)
        print(args.out_dir)
        print("Epoch {}: Acc (%): {:.3f}\t".format(epoch, id_acc))
        
        if best_id_acc < id_acc:
            torch.save(net.state_dict(), os.path.join(args.out_dir, 'bestpoint.pth.tar'))
            best_id_acc = id_acc
    print(best_id_acc)
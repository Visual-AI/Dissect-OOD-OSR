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
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

from data.open_set_datasets import get_class_splits, get_datasets
from methods.OOD.util.data_loader import get_loader_in, get_loader_out

from methods import train_oe
from utils.utils import seed_torch, str2bool
from utils.schedulers import get_scheduler, WarmUpLR
from utils.yfcc_ImageFolder import ImageFolder
import timm
import sys
from os import path

parser = argparse.ArgumentParser(description='Finerue a ImageNet Classifier with OE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir',type=str, default='/disk/datasets/ood_zoo',help='dir of output')
parser.add_argument('--yfcc-dir',type=str, default='/disk/datasets',help='dir of output')

parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')
parser.add_argument('--config', nargs="?", type=str, default="/home/hjwang/osrd/train_configs.yaml", help="Configuration file to use")

parser.add_argument('--in-dataset', type=str, default='imagenet', help='Choose between cifar-10-10, cifar-10-100-10, cifar-10-100-50, tinyimagenet')
parser.add_argument('--model', type=str, default='resnet50', help='Choose architecture.')
parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim, only for classifier32 at the moment")

# Optimization options
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, help='The initial learning rate.')
parser.add_argument('--lamb', type=float, help='The balance factor.')

parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 penalty).')
parser.add_argument('--batch_size', type=int, default=96, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=150, help='Batch size.')
parser.add_argument('--scheduler', type=str)

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--temp', type=float, default=1.0, help="temp")

parser.add_argument('--loss', type=str, default='OELoss')
parser.add_argument('--loss_strategy', default='OE', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--ablation', type=str, help='conv-default/oe-default')

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

args.epochs = 5
args.scheduler = "cosine_warm_restarts_warmup"
args.optim = "sgd"
args.weight_decay = 5.0e-4
args.is_nesterov = True
args.steps, args.gamma = None, None
args.rand_aug_m, args.rand_aug_n = None, None
args.label_smoothing = None

args.ablation = 'OOD'
args.model = 'resnet50'

# corruption_imgs = [('/disk/datasets/clip_data/26/c066d96a9e9aa9eff628b7935747019.jpg', 26), ('/disk/datasets/clip_data/27/c0672cd8cab8c654a79ca22ca513e89.jpg', 27), ('/disk/datasets/clip_data/15/c0671f161ff5c57e22a86188b6ab623.jpg', 15), ('/disk/datasets/clip_data/32/c06717543dc734e2871927f1be57b876.jpg', 32), ('/disk/datasets/clip_data/3/c066c249f28a8215eac4a97947f59e.jpg', 3), ('/disk/datasets/clip_data/42/c06711db77f7d814a96455c3545b918.jpg', 42), ('/disk/datasets/clip_data/0/c760fa44567edfe6b4d2f9ed41dab6.jpg', 0), ('/disk/datasets/clip_data/47/c066c45ef122b2cdecb04452ceebaf61.jpg', 47), ('/disk/datasets/clip_data/47/c066e721f80af786b45e4149533422.jpg', 47), ('/disk/datasets/clip_data/39/c066be9cd9cc65e7f72eea588158e38d.jpg', 39), ('/disk/datasets/clip_data/39/22d0fadda7a5faa729e1883b42715f.jpg', 39), ('/disk/datasets/clip_data/35/c760eaec5c6dc5ac401e33443aade3f3.jpg', 35), ('/disk/datasets/clip_data/46/c066ee5094dedf2109c93ffc9f71e1.jpg', 46), ('/disk/datasets/clip_data/7/c067146408f43a73a8712219e1e954.jpg', 7), ('/disk/datasets/clip_data/7/c760d1e0c92dfa77787548862af75fed.jpg', 7), ('/disk/datasets/clip_data/23/c7612144e4a5cad23e8b60c0419fc923.jpg', 23), ('/disk/datasets/clip_data/38/c066f4f1adfb42213027887d293735.jpg', 38), ('/disk/datasets/clip_data/24/c066e1c6f1c36e58ee68e466d576837.jpg', 24)]
corruption_imgs = ['/disk/datasets/clip_data/26/c066d96a9e9aa9eff628b7935747019.jpg', '/disk/datasets/clip_data/7/c760d1e0c92dfa77787548862af75fed.jpg', '/disk/datasets/clip_data/27/c0672cd8cab8c654a79ca22ca513e89.jpg', '/disk/datasets/clip_data/15/c0671f161ff5c57e22a86188b6ab623.jpg', '/disk/datasets/clip_data/32/c06717543dc734e2871927f1be57b876.jpg', '/disk/datasets/clip_data/3/c066c249f28a8215eac4a97947f59e.jpg', '/disk/datasets/clip_data/42/c06711db77f7d814a96455c3545b918.jpg', '/disk/datasets/clip_data/0/c760fa44567edfe6b4d2f9ed41dab6.jpg', '/disk/datasets/clip_data/47/c066c45ef122b2cdecb04452ceebaf61.jpg', '/disk/datasets/clip_data/47/c066e721f80af786b45e4149533422.jpg', '/disk/datasets/clip_data/39/c066be9cd9cc65e7f72eea588158e38d.jpg', '/disk/datasets/clip_data/39/22d0fadda7a5faa729e1883b42715f.jpg', '/disk/datasets/clip_data/35/c760eaec5c6dc5ac401e33443aade3f3.jpg', '/disk/datasets/clip_data/46/c066ee5094dedf2109c93ffc9f71e1.jpg', '/disk/datasets/clip_data/7/c067146408f43a73a8712219e1e954.jpg', '/disk/datasets/clip_data/7/c760d1e0c92dfa77787548862af75fed.jpg', '/disk/datasets/clip_data/23/c7612144e4a5cad23e8b60c0419fc923.jpg', '/disk/datasets/clip_data/38/c066f4f1adfb42213027887d293735.jpg', '/disk/datasets/clip_data/24/c066e1c6f1c36e58ee68e466d576837.jpg']

def get_model(num_classes):
    # Create model
    if args.model == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model)
    
    model = model.to(args.device)
    return model


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
                logits = net(data)
                total += labels.size(0)
                correct += (logits.data.max(1)[1] == labels.data).sum()

        id_acc = float(correct) * 100. / float(total)

    return id_acc


def get_optimizer(params_list, weight_decay=None):
    optimizer = torch.optim.Adam(params_list, lr=args.lr) 
    return optimizer


if __name__ == '__main__':
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

    # yfcc_dataset = ImageFolder(os.path.join(args.yfcc_dir, 'clip_data'), transform=transform_train_largescale)
    # yfcc_dataset.imgs = [i for i in yfcc_dataset.imgs if not i[0] in corruption_imgs]
    # oeloader = torch.utils.data.DataLoader(yfcc_dataset, batch_size=args.oe_batch_size, shuffle=True)

    oeloader = torch.utils.data.DataLoader(
                    torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'imagenet-r'), transform_train_largescale),
                    batch_size=args.batch_size, shuffle=True)

    # GET LOSS
    criterion = getattr(importlib.import_module('methods.loss.'+'Softmax'), 'Softmax')(**options)

    # Get base network
    net = get_model(num_classes=options['num_classes'])

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
        train_oe(net, optimizer, scheduler, warmup_scheduler, trainloader, oeloader, epoch, optimizer_h=optimizer_h, **options)
        id_acc = test_id(net, testloader, **options)
        print(args.out_dir)
        print("Epoch {}: Acc (%): {:.3f}\t".format(epoch, id_acc))
        
        if best_id_acc < id_acc:
            torch.save(net.state_dict(), os.path.join(args.out_dir, 'bestpoint.pth.tar'))
            best_id_acc = id_acc
    print(best_id_acc)
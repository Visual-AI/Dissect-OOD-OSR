# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
CUDA_VISIBLE_DEVICES=0 python3 train_vit.py --model vit_small --config_strategy 'bas_config' --loss_strategy 'CE'
'''

from __future__ import print_function
import sys
   
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
import importlib

from vit import ViT
from convmixer import ConvMixer
sys.path.append("..")
from data.augmentations.randaugment import RandAugment
from utils.utils import seed_torch, str2bool, load_networks, save_networks
from utils.schedulers import get_scheduler

from data.open_set_datasets import get_class_splits, get_datasets
from methods.OOD.util.data_loader import get_loader_in, get_loader_out

from methods import train_cs
from methods.ARPL.arpl_models import gan

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--model', default='vit_small', type=str)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")

parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")

parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim")

parser.add_argument('--in-dataset', type=str, default='cifar-10-100-10', help="cifar-10-10/cifar-10-100-10/cifar-10-100-50/tinyimagenet")
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')

parser.add_argument('--config_strategy', default='bas_config', type=str, help="bas_config/mls_config")
parser.add_argument('--loss_strategy', default='ARPL_CS', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--ablation', type=str, default='.', help='ablation study')

args = parser.parse_args()
device = args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.image_size = 64 if args.in_dataset == 'tinyimagenet' else 32

if args.in_dataset in ['cifar-10', 'cifar-100']:
    args.is_ood = True
else:
    args.is_ood = False

if 'cifar-10-100' in args.in_dataset:
    args.out_num = int(args.in_dataset.split('-')[-1])
    args.in_dataset = 'cifar-10-100'

if args.config_strategy == 'bas_config':
    args.epochs = 200
    # args.scheduler = 'multi_step'
    # args.steps = [30, 60, 90, 120]
    args.batch_size, args.oe_batch_size = 128, 256

    if args.in_dataset == 'tinyimagenet':
        args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0.9, 1, 9
        args.batch_size, args.oe_batch_size = 64, 128
        # args.optim = 'adam'
        args.lr = 1e-4
    else:
        args.lr = 1e-4
        args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0, 1, 6

elif args.config_strategy == 'mls_config':
    args.epochs = 600
    args.scheduler = 'cosine_warm_restarts_warmup'
    args.optim = 'sgd'
    args.batch_size, args.oe_batch_size = 128, 256
    args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0, 1, 6
    
    if args.in_dataset == 'tinyimagenet':
        args.batch_size, args.oe_batch_size = 64, 128
        args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0.9, 1, 9
        args.optim = 'adam'
        args.lr = 1e-4
    elif args.in_dataset == 'cifar-10-100':
        args.lr = 1e-4
    else:
        args.lr = 1e-4

if args.transform == 'rand-augment' and args.rand_aug_m is not None and args.rand_aug_n is not None:
    args.ablation = '{}_{}_{}_{}'.format(args.transform, args.rand_aug_m, args.rand_aug_n, args.ablation)
if args.label_smoothing is not None:
    args.ablation = 'Smoothing{}_{}'.format(args.label_smoothing, args.ablation)
args.ablation = '{}_{}_{}_{}_{}'.format(args.in_dataset, args.model, args.loss_strategy, args.config_strategy, args.ablation)
args.out_dir = os.path.join(args.out_dir, args.ablation)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print(args.out_dir)

options = vars(args)

print("Creating GAN")
nz, ns = args.nz, 1
if args.image_size >= 64:
    netG = gan.Generator(1, nz, 64, 3)
    netD = gan.Discriminator(1, 3, 64)
else:
    netG = gan.Generator32(1, nz, 64, 3)
    netD = gan.Discriminator32(1, 3, 64)
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
criterionD = nn.BCELoss()
netG = netG.to(device)
netD = netD.to(device)
fixed_noise.to(device)


optimizerD = torch.optim.Adam(netD.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))

# use cosine or reduce LR on Plateau scheduling
# if not args.cos:
#     from torch.optim import lr_scheduler
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
# else:
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

# take in args
import wandb
watermark = "{}_lr{}".format(args.model, args.lr)
if args.amp:
    watermark += "_useamp"

wandb.init(project="cifar10-challange", name=watermark)
wandb.config.update(args)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.model=="vit_timm":
    size = 384
else:
    size = args.image_size

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
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=shuffle, sampler=None, num_workers=16)

    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    options.update(
        {
            'known':    args.train_classes,
            'unknown':  args.open_set_classes,
            'dataloaders': dataloaders,
            'num_classes': len(args.train_classes),
            'feat_dim': 512
        }
    )

oeloader = None

# GET LOSS
Loss = importlib.import_module('methods.loss.'+args.loss)
criterion = getattr(Loss, args.loss)(**options)
criterion = criterion.to(args.device)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# model = VGG('VGG19')
if args.model=="vit_small":
    if args.loss_strategy == 'ARPL_CS':
        from models.vit_small_cs import ViT_cs
        model = ViT_cs(
        image_size = size,
        patch_size = args.patch,
        num_classes = len(args.train_classes) if not args.is_ood else len(classes),
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    else:
        from models.vit_small import ViT
        model = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = len(args.train_classes),
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.model=="vit":
    # ViT for cifar10
    model = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.model=="vit_timm":
    import timm
    model = timm.create_model("vit_base_patch16_384", pretrained=True)
    model.head = nn.Linear(model.head.in_features, 10)
elif args.model=="cait":
    from models.cait import CaiT
    model = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.model=="cait_small":
    from models.cait import CaiT
    model = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.model=="swin":
    from models.swin import swin_t
    model = swin_t(window_size=args.patch, num_classes=10, downscaling_factors=(2,2,2,1))

if device == 'cuda':
    model = torch.nn.DataParallel(model) # make parallel
    cudnn.benchmark = True

if args.opt == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr)  

if args.config_strategy == 'bas_config':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
else:
    scheduler = get_scheduler(optimizer, args)

try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    term_width = 80

term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, tuples in enumerate(trainloader):
        if len(tuples) == 2:
            data, labels = tuples
        elif len(tuples) == 3:
            data, labels, idx = tuples

        data, labels = data.to(device), labels.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=args.amp):
            x, y = model(data, return_feat=True)
            logits, loss = criterion(x, y, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, tuples in enumerate(testloader):
            if len(tuples) == 2:
                inputs, targets = tuples
            elif len(tuples) == 3:
                inputs, targets, idx = tuples

            inputs, targets = inputs.to(device), targets.to(device)
            x, y = model(inputs, return_feat=True)
            logits, loss = criterion(x, y, targets)

            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        save_networks(model, args.out_dir, 'bestpoint.pth.tar', options, criterion=criterion)
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.model}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

wandb.watch(model)
for epoch in range(start_epoch, args.epochs):
    start = time.time()

    train_cs(model, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader, epoch=epoch, **options)
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    # STEP SCHEDULER
    if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
        scheduler.step(acc, epoch)
    elif args.scheduler == 'multi_step':
        scheduler.step()
    else:
        scheduler.step(epoch=epoch)

    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"], "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.model}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

# writeout wandb
wandb.save("wandb_{}.h5".format(args.model))
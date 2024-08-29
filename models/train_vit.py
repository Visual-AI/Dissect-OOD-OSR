# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
CUDA_VISIBLE_DEVICES=0 python3 train_vit.py --model vit_small --config_strategy 'bas_config' --loss_strategy 'CE' --is_ood
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

from convmixer import ConvMixer
sys.path.append("..")
from data.augmentations.randaugment import RandAugment
from data.open_set_datasets import get_class_splits, get_datasets
from utils.utils import seed_torch, str2bool, load_networks, save_networks
from utils.schedulers import get_scheduler
from utils.tinyimages_80mn_loader import TinyImages
# from methods.ARPL.core import train_cs
# from methods.ARPL.arpl_models import gan

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data-dir',type=str, default='/disk/datasets/ood_zoo', help='dir of output')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--model', default='vit_small', type=str)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--loss', type=str, default='Softmax')
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
parser.add_argument('--loss_strategy', default='', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--is_ood', action='store_true', default=False, help='running ood/osr')
parser.add_argument('--ablation', type=str, default='.', help='ablation study')

args = parser.parse_args()
device = args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.is_ood:
    args.in_dataset = 'cifar-10'
    args.num_classes = 10
else:
    if args.in_dataset == 'tinyimagenet':
        args.image_size = 64
    elif 'cifar-10-100' in args.in_dataset:
        args.out_num = int(args.in_dataset.split('-')[-1])
        args.in_dataset = 'cifar-10-100'
    else:
        args.in_dataset = 'cifar-10-10'

if args.config_strategy == 'bas_config':
    args.epochs = 200
    # args.scheduler = 'multi_step'
    # args.steps = [30, 60, 90, 120]
    args.batch_size, args.oe_batch_size = 128, 256

    if args.in_dataset == 'tinyimagenet':
        args.label_smoothing, args.rand_aug_n, args.rand_aug_m = 0.9, 1, 9
        args.batch_size, args.oe_batch_size = 64, 128
        if 'ARPL' in args.loss_strategy:
            # args.optim = 'adam'
            args.lr = 1e-4
        else:
            args.lr = 1e-3
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
        if 'ARPL' in args.loss_strategy:
            args.optim = 'adam'
            args.lr = 1e-4
        else:
            args.lr = 1e-3
    elif args.in_dataset == 'cifar-10-100':
        args.lr = 1e-4
    else:
        args.lr = 1e-4
        if not 'ARPL' in args.loss_strategy:
            args.rand_aug_m = 15

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

if args.loss_strategy == 'ARPL_CS':
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


if args.loss_strategy == 'ARPL_CS':
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))

# use cosine or reduce LR on Plateau scheduling
# if not args.cos:
#     from torch.optim import lr_scheduler
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
# else:
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

# take in args
watermark = "{}_lr{}".format(args.model, args.lr)
if args.amp:
    watermark += "_useamp"


bs = args.batch_size
use_amp = args.amp

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.model=="vit_timm":
    size = 384
else:
    size = args.image_size

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if args.is_ood:
    if args.transform == 'rand-augment':
        if args.rand_aug_m is not None:
            if args.rand_aug_n is not None:
                transform_train.transforms.insert(0, RandAugment(args.rand_aug_m, args.rand_aug_n, args=args))

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'id_data'), train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'id_data'), train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8) 
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
        dataloaders[k] = DataLoader(v, batch_size=bs, shuffle=shuffle, sampler=None, num_workers=16)

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

if args.loss_strategy == 'OE':
    if args.in_dataset == 'tinyimagenet':
        ood_data = TinyImages(transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Resize((args.image_size, args.image_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
    else:
        ood_data = TinyImages(transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
    oeloader = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=False, num_workers=4, pin_memory=True)
else:
    oeloader = None

# GET LOSS
if 'ARPL' in args.loss_strategy:
    args.loss = "ARPLoss"
elif args.loss_strategy == 'OE':
    args.loss = "OELoss"
Loss = importlib.import_module('methods.loss.'+args.loss)
criterion = getattr(Loss, args.loss)(**options)
criterion = criterion.to(args.device)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# model = VGG('VGG19')
if args.model=='res18':
    model = ResNet18()
elif args.model=='vgg':
    model = VGG('VGG19')
elif args.model=='res34':
    model = ResNet34()
elif args.model=='res50':
    model = ResNet50()
elif args.model=='res101':
    model = ResNet101()
elif args.model=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    model = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.model=="vit_small":
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
        num_classes = len(args.train_classes) if not args.is_ood else len(classes),
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

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.model))
#     model.load_state_dict(checkpoint['model'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

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
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    if options['loss_strategy'] == 'OE':
        oeloader.dataset.offset = np.random.randint(len(oeloader.dataset))
        batch_idx = 0
        for in_tuples, out_tuples in zip(trainloader, oeloader):
            batch_idx += 1
            if len(in_tuples) == 2:
                in_data, in_labels = in_tuples
                out_data, out_labels = out_tuples
            elif len(in_tuples) == 3:
                in_data, in_labels, idx = in_tuples            
                out_data, out_labels = out_tuples

            data = torch.cat((in_data, out_data), 0)            
            data, in_labels = data.to(device), in_labels.to(device)

            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                x, y = model(data, return_feat=True)
                logits, loss = criterion(x, y, in_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = logits[:len(in_labels)].max(1)
            total += in_labels.size(0)
            correct += predicted.eq(in_labels).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    else:
        for batch_idx, tuples in enumerate(trainloader):
            if len(tuples) == 2:
                data, labels = tuples
            elif len(tuples) == 3:
                data, labels, idx = tuples

            data, labels = data.to(device), labels.to(device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = model(data)
                logits, loss = criterion(x, x, labels)
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
        if 'ARPL' in args.loss_strategy:
            save_networks(model, args.out_dir, 'bestpoint.pth.tar', options, criterion=criterion)
        else:
            state = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scaler": scaler.state_dict()}
            torch.save(state, os.path.join(args.out_dir, 'bestpoint.pth.tar'))
        
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.model}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

for epoch in range(start_epoch, args.epochs):
    start = time.time()

    if args.loss_strategy == 'ARPL_CS':
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
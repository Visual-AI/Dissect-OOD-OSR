import os
import sys
import argparse
import datetime
import time
import csv
import os.path as osp
import numpy as np
import warnings
import importlib
warnings.filterwarnings('ignore')
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.utils as vutils
from utils.tinyimages_80mn_loader import TinyImages
import torchvision.transforms as transforms
import timm 

from methods.ARPL.arpl_models.resnetABN import resnet18ABN
from methods.ARPL.arpl_models import gan
from methods.ARPL.core import train, train_cs, test
from methods.OOD.util.data_loader import get_loader_in, get_loader_out
from utils.utils import seed_torch, str2bool, load_networks, save_networks
from utils.schedulers import get_scheduler

parser = argparse.ArgumentParser("ARPLoss")

# dataset
parser.add_argument('--data-dir',type=str, default='/home/hjwang/osrd/methods/OOD/datasets',help='dir of output')
parser.add_argument('--in-dataset', type=str, default='cifar-10')
parser.add_argument('--out-datasets', default=['SVHN', 'dtd', 'LSUN', 'LSUN_R', 'iSUN', 'places365'], type=list, help="['SVHN', 'LSUN', 'LSUN_R', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd']")
parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")

# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")

# model
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim")
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")

# aug
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--rand_aug_m', type=int, default=None)

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resume',type=str2bool, default=False, help='whether to resume training')
parser.add_argument('--resume-dir',type=str, default='', help='dir of output')

parser.add_argument('--ood_method', default='', type=str, help='mls/odin/energy/react')
parser.add_argument('--loss_strategy', default='', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--config_strategy', default='bas_config', type=str, help="bas_config/mls_config")
parser.add_argument('--ablation', type=str, default='.', help='ablation study')
parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.config_strategy == 'bas_config':
    args.epochs = 100
    args.scheduler = 'multi_step'
    if 'ARPL' in args.loss_strategy:
        args.batch_size = 128
        args.optim, args.lr = 'adam', 0.0001
        args.steps = [30, 60, 90, 120]
    else:
        args.batch_size, args.oe_batch_size = 64, 128
        args.optim, args.lr = 'sgd', 0.1

elif args.config_strategy == 'mls_config':
    args.epochs = 600
    args.scheduler = 'cosine_warm_restarts_warmup'
    args.rand_aug_n, args.rand_aug_m, args.label_smoothing = 1, 15, 0
    args.batch_size, args.oe_batch_size = 128, 256
    if 'ARPL' in args.loss_strategy:
        args.optim, args.lr = 'adam', 0.0001
    else:
        args.optim, args.lr = 'sgd', 0.1

if args.loss_strategy == 'OE':
    args.epochs = 100
    # args.model = 'wrn'
    args.batch_size, args.oe_batch_size = 128, 256

if args.transform == 'rand-augment' and args.rand_aug_m is not None and args.rand_aug_n is not None:
    args.ablation = '{}_{}_{}_{}'.format(args.transform, args.rand_aug_m, args.rand_aug_n, args.ablation)
if args.label_smoothing is not None:
    args.ablation = 'Smoothing{}_{}'.format(args.label_smoothing, args.ablation)
args.ablation = '{}_{}_{}_{}_{}'.format(args.in_dataset, args.model, args.loss_strategy, args.config_strategy, args.ablation)
args.out_dir = os.path.join(args.out_dir, args.ablation)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
print(args.out_dir)


def cal_training_stats(net, trainloader, options):
    net.eval()
    tr_pbar = tqdm(trainloader)
    stats = 0
    total = 0

    for batch_idx, tuples in enumerate(tr_pbar):
        if len(tuples) == 2:
            data, labels = tuples
        elif len(tuples) == 3:
            data, labels, idx = tuples
        
        total += data.size(0)
        data, labels = data.to(options['device']), labels.to(options['device'])

        with torch.no_grad():
            act = net.features(data, which_layer=1)
            B, c = act.size()[:2]
            act_flat = act.view(B, -1)
            act_max = act_flat.max(dim=-1)[0]
            act_max_sum = torch.sum(act_max, 0)

            if batch_idx == 0:
                stats = act_max_sum
            else:
                stats += act_max_sum
    print(stats / total)
    return stats / total

def get_optimizer(args, params_list):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=args.lr)
    return optimizer

def get_model(num_classes):
    if args.model == 'resnet18':
        from models.resnet import resnet18_cifar
        model = resnet18_cifar(num_classes=num_classes)
    elif args.model == 'wrn':
        from models.wrn import WideResNet
        model = WideResNet(40, num_classes, 2, dropRate=0.3)
    elif args.model == 'densnet121':
        from models.densenet import densenet121
        model = densenet121(num_classes=num_classes)
    elif args.model == 'vit':
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    else:
        assert False, 'Not supported model arch: {}'.format(args.model)
    return model

def main():
    if args.in_dataset == 'imagenet':
        loader_in_dict = get_loader_in(args, split=('val'))
        testloaderIn, args.num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    else:
        loader_in_dict = get_loader_in(args)
        trainloader, testloaderIn, args.num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes 

    if args.loss_strategy == 'OE':
        ood_data = TinyImages(os.path.join(args.data_dir, '300K_random_images.npy'), transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
        oeloader = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        oeloader = None

    if 'CS' in args.loss_strategy:
        net = resnet18ABN(num_classes=args.num_classes, num_bns=2)
    else:
        net = get_model(args.num_classes)
    net = net.to(args.device)

    if 'ARPL' in args.loss_strategy:
        args.loss = "ARPLoss"
    elif args.loss_strategy == 'OE':
        args.loss = "OELoss"    

    options = vars(args)

    if options['ood_method'] == 'ours':
        options['mean'] = cal_training_stats(net, trainloader, options)
        options['mean'] = cal_training_stats2(net, trainloader, options)

    Loss = importlib.import_module('methods.ARPL.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)
    criterion = criterion.to(args.device)

    if 'CS' in args.loss_strategy:
        print("Creating GAN")
        nz = options['nz']
        netG = gan.Generator32(1, nz, 64, 3) # ngpu, nz, ngf, nc
        netD = gan.Discriminator32(1, 3, 64) # ngpu, nc, ndf
        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()
        netG = netG.to(args.device)
        netD = netD.to(args.device)
        fixed_noise.to(args.device)

    
    if args.loss_strategy == 'OE':
        optimizer = torch.optim.SGD(net.parameters(), 0.001, 0.9, weight_decay=0.0005, nesterov=True)
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(testloaderIn), 1, 1e-6 / 0.001))
    else:
        params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
        optimizer = get_optimizer(args=args, params_list=params_list)
        scheduler = get_scheduler(optimizer, args)

    options['scheduler'] = scheduler

    if 'CS' in args.loss_strategy:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
 
    best_avg_ACC = 0

    if args.resume:
        if 'ARPL' in args.loss_strategy:
            if os.path.exists(os.path.join(args.out_dir, 'bestpoint.pth.tar')):
                net, criterion = load_networks(net, args.out_dir, 'bestpoint.pth.tar', options, criterion=criterion)
            elif os.path.exists(args.resume_dir):
                net, criterion = load_networks(net, args.resume_dir, '', options, criterion=criterion)
        elif args.model == 'vit':
            pass
        else:
            if os.path.exists(args.resume_dir):
                net.load_state_dict(torch.load(args.resume_dir))
            else:
                net.load_state_dict(torch.load(os.path.join(args.out_dir, 'bestpoint.pth.tar')))

        for out_dataset in args.out_datasets:
            testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader
            ans = test(net, criterion, testloaderIn, testloaderOut, epoch=0, **options)
            print(ans)

    else:
        for epoch in range(options['epochs']):
            print("==> Epoch {}/{}".format(epoch+1, options['epochs']))

            if 'CS' in args.loss_strategy:
                train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader, epoch=epoch, **options)

            train(net, criterion, optimizer, trainloader, oeloader, epoch=epoch, **options)

            print("==> Test")
            for out_dataset in args.out_datasets:
                testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader
                ans = test(net, criterion, testloaderIn, testloaderOut, epoch=0, **options)
                print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(ans['ACC'], ans['AUROC'], ans['OSCR']))
                if not epoch == options['epochs'] - 1:
                    break
            
            if ans['ACC'] > best_avg_ACC:
                print("Saved the best checkpoint.")
                best_avg_ACC = ans['ACC']
                if 'ARPL' in args.loss_strategy:
                    save_networks(net, args.out_dir, 'bestpoint.pth.tar', options, criterion=criterion)
                else:
                    torch.save(net.state_dict(), os.path.join(args.out_dir, 'bestpoint.pth.tar'))

            if not args.loss_strategy == 'OE':
                if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
                    scheduler.step(ans['ACC'], epoch)
                elif args.scheduler == 'multi_step':
                    scheduler.step()
                else:
                    scheduler.step(epoch=epoch)


if __name__ == '__main__':
    main()
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

from methods.ARPL.arpl_models import gan
from methods.ARPL.arpl_models.arpl_models import classifier32ABN
from methods.ARPL.arpl_models.wrapper_classes import TimmResNetWrapper
from methods.ARPL.arpl_utils import save_networks
from methods.ARPL.core import train, train_cs, test

from utils.utils import seed_torch, str2bool
from methods.ARPL.init_hypers import get_default_hyperparameters
from utils.schedulers import get_scheduler
from data.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model
from torch.utils.tensorboard import SummaryWriter

from config import exp_root

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub', help="")
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=64)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str)
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco', help='Which pretraining to use if --model=timm_resnet50_pretrained. Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

# misc
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--train_feat_extractor', default=True, type=str2bool, help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')

parser.add_argument('--eval_strategy', default='logit', type=str, help="logit/msp/odin/mahalanobis/energy/gradnorm")
parser.add_argument('--is_pretrained', default=False, type=str2bool, help='Do we use ImageNet pretrained model', metavar='BOOL')
parser.add_argument('--ablation', type=str, default='', help='ablation study')


def get_optimizer(args, params_list):
    if args.optim is None:
        optimizer = torch.optim.Adam(params_list, lr=args.lr) if options['dataset'] == 'tinyimagenet' else torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=args.lr)
    return optimizer


# TODO: Args and options are largely duplicates: tidy up
def main_worker(options, args):
    model_path = os.path.join(args.log_dir, args.ablation)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cudnn.benchmark = False
    torch.cuda.manual_seed_all(options['seed'])

    # DATALOADERS
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    # MODEL
    print("Creating model: {}".format(options['model']))
    if options['cs'] and args.loss == 'ARPLoss':
        if args.model == 'classifier32':
            net = classifier32ABN(num_classes=len(args.train_classes), feat_dim=args.feat_dim)

        nz, ns = options['nz'], 1
        if args.image_size >= 64:
            netG = gan.Generator(1, nz, 64, 3)
            netD = gan.Discriminator(1, 3, 64)
        else:
            netG = gan.Generator32(1, nz, 64, 3)
            netD = gan.Discriminator32(1, 3, 64)
        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()
        
        netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).to(device)
        netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).to(device)
        fixed_noise.to(device)

        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))

    else:
        wrapper_class = TimmResNetWrapper if args.model == 'timm_resnet50_pretrained' else None
        net = get_model(args, wrapper_class=wrapper_class)

    # Loss
    options.update(
        {
            'feat_dim': args.feat_dim,
            'use_gpu':  device
        }
    )

    # GET LOSS
    Loss = importlib.import_module('methods.ARPL.loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)
    test_Loss = importlib.import_module('methods.ARPL.loss.'+'Softmax')
    test_criterion = getattr(test_Loss, 'Softmax')(**options)
    
    # PREPARE EXPERIMENT
    net = net.to(device)
    # net = nn.DataParallel(net).to(device)
    criterion = criterion.to(device)
    params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=params_list)
    # GET SCHEDULER
    scheduler = get_scheduler(optimizer, args)

    best_AUROC, best_results = 0, 0

    # TRAIN
    for epoch in range(options['epochs']):
        if options['cs'] and args.loss == 'ARPLoss':
            train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader, epoch=epoch, **options)
        else:
            train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['epochs']:
            results = test(net, test_criterion, testloader, outloader, epoch=epoch, **options)
            print("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t".format(epoch, results['ACC'], results['AUROC']))
            if best_AUROC < results['AUROC']:
                save_networks(net, model_path, 'best', options['loss'])
                results['Info'] = "The best model at {}-th epoch".format(epoch)
                best_AUROC = results['AUROC']
                best_results = results

            # LOG
            args.writer.add_scalar('Test Acc Top 1', results['ACC'], epoch)
            args.writer.add_scalar('AUROC', results['AUROC'], epoch)

        # STEP SCHEDULER
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results['ACC'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)

    save_networks(net, model_path, 'last', options['loss'])

    return best_results


if __name__ == '__main__':
    args = parser.parse_args()
    args = get_default_hyperparameters(args)

    args.ablation = args.dataset + '_' + args.model + '_' + args.ablation
    results = dict()

    exp_root = os.path.join(exp_root, args.ablation)
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    args.log_dir = os.path.join(exp_root, 'logs')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    args.writer = SummaryWriter(log_dir=os.path.join(exp_root, 'tfboard'))
    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # SEED
    seed_torch(args.seed)
    # INIT
    if args.feat_dim is None:
        args.feat_dim = 128 if args.model == 'classifier32' else 2048

    # DATASETS
    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx, cifar_plus_n=args.out_num)
    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
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
        batch_size = 1 if not k == 'train' and args.eval_strategy =='gradnorm' else args.batch_size
        dataloaders[k] = DataLoader(v, batch_size=batch_size, shuffle=shuffle, sampler=None, num_workers=16)

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

    if options['dataset'] == 'cifar-10-100':
        file_name = '{}_{}_{}_{}.csv'.format(options['dataset'], options['out_num'], options['model'], options['loss'])
        if options['cs']:
            file_name = '{}_{}_{}_{}_cs.csv'.format(options['dataset'], options['out_num'], options['model'], options['loss'])
    else:
        file_name = '{}_{}_{}.csv'.format(options['dataset'], options['model'], options['loss'])
        if options['cs']:
            file_name = '{}_{}_{}_cs.csv'.format(options['dataset'], options['model'], options['loss'])

    # TRAIN
    res = main_worker(options, args)

    # LOG
    res['split_idx'] = args.split_idx
    res['unknown'] = args.open_set_classes
    res['known'] = args.train_classes
    res['ID'] = args.log_dir.split("/")[-1]
    results[str(args.split_idx)] = res
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(dir_path, file_name), mode='a', header=False)
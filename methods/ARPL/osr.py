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
from tqdm import tqdm

from methods.ARPL.arpl_models import gan
from methods.ARPL.arpl_models.resnetABN import resnet18ABN
from methods.ARPL.arpl_models.wrapper_classes import TimmResNetWrapper
from methods.ARPL.arpl_utils import save_networks
from methods import train, train_cs, test

from utils.utils import seed_torch, str2bool
from utils.tinyimages_80mn_loader import TinyImages

from methods.ARPL.init_hypers import get_default_hyperparameters
from utils.schedulers import get_scheduler
from data.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--loss', type=str, default='Softmax')
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
parser.add_argument('--resume-dir',type=str, default='/disk1/hjwang/ssb_models_cub/cub_599_Softmax.pth', help='dir of output')

parser.add_argument('--ood_method', default='mls', type=str, help="mls/odin/energy/react")
parser.add_argument('--loss_strategy', default='', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--config_strategy', default='bas_config', type=str, help="bas_config/mls_config")
parser.add_argument('--is_pretrained', default=False, type=str2bool, help='Do we use ImageNet pretrained model', metavar='BOOL')
parser.add_argument('--ablation', type=str, default='.', help='ablation study')

args = parser.parse_args()
args.image_size = 64 if args.in_dataset == 'tinyimagenet' else 32

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
        if 'ARPL' in args.loss_strategy:
            args.optim = 'adam'
            args.lr = 0.001
        else:
            args.lr = 0.01
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
        if 'ARPL' in args.loss_strategy:
            args.optim = 'adam'
            args.lr = 0.001  
        else:
            args.lr = 0.01
    elif args.in_dataset == 'cifar-10-100':
        args.lr = 0.1
    else:
        args.lr = 0.1
        if not 'ARPL' in args.loss_strategy:
            args.rand_aug_m = 15

# if args.model == 'vit_small'
#     args.lr = 1e-3 
#     args.optim = 'adam'

def get_optimizer(args, params_list):
    if args.optim is None:
        optimizer = torch.optim.Adam(params_list, lr=args.lr) if args.in_dataset == 'tinyimagenet' else torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=args.lr)
    return optimizer


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


def main_worker(options, args):
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

    # DATALOADERS
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    if args.loss_strategy == 'OE':
        if args.in_dataset == 'tinyimagenet':
            ood_data = TinyImages(transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Resize((args.image_size, args.image_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
        else:
            ood_data = TinyImages(transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
        oeloader = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        oeloader = None

    # MODEL
    print("Creating model: {}".format(args.model))
    if args.model == 'resnet18':
        net = get_model(args)
    elif args.model == 'timm_resnet50_pretrained':
        wrapper_class = TimmResNetWrapper
        net = get_model(args)
    elif args.model == 'vit_small':
        from models.vit_small import ViT
        net = ViT(image_size=args.image_size, patch_size=4, num_classes=len(args.train_classes), dim=512, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    else:
        net = get_model(args)

    net = net.to(args.device)
    # net = nn.DataParallel(net).to(args.device)
    if options['ood_method'] == 'ours':
        options['mean'] = cal_training_stats(net, trainloader, options)

    # -----------------------------
    # --CS MODEL AND LOSS
    # -----------------------------
    if args.loss_strategy == 'ARPL_CS':
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
    if 'ARPL' in args.loss_strategy:
        args.loss = "ARPLoss"
    elif args.loss_strategy == 'OE':
        args.loss = "OELoss"
    Loss = importlib.import_module('methods.ARPL.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)
    

    criterion = criterion.to(args.device)
    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=[{'params': net.parameters()}, {'params': criterion.parameters()}])

    if args.loss_strategy == 'ARPL_CS':
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
        fixed_noise.to(args.device)

    # GET SCHEDULER
    scheduler = get_scheduler(optimizer, args)

    best_AUROC, best_avg_ACC, best_results = 0, 0, 0

    if args.resume:
        if not args.model == 'timm_resnet50_pretrained':
            if args.model == 'vit_small':
                net = torch.nn.DataParallel(net)
                net.load_state_dict(torch.load(os.path.join(args.out_dir, 'bestpoint.pth.tar'))['model'])
            else:
                if 'ARPL' in args.loss_strategy:
                    if os.path.exists(os.path.join(args.out_dir, 'bestpoint.pth.tar')):
                        net, criterion = load_networks(net, args.out_dir, 'bestpoint.pth.tar', options, criterion=criterion)
                    elif os.path.exists(args.resume_dir):
                        net, criterion = load_networks(net, args.resume_dir, '', options, criterion=criterion)

                else:
                    if os.path.exists(args.resume_dir):
                        net.load_state_dict(torch.load(args.resume_dir))
                    else:
                        net.load_state_dict(torch.load(os.path.join(args.out_dir, 'bestpoint.pth.tar')))
        best_results = test(net, criterion, trainloader, testloader, outloader, **options)

    else:    
        # TRAIN
        for epoch in range(args.epochs):
            if args.loss_strategy == 'ARPL_CS':
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

    return best_results


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

    args.writer = SummaryWriter(log_dir=os.path.join(exp_root, 'tfboard'))
    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    # SEED
    seed_torch(args.seed)
    # INIT
    if args.model == 'classifier32':
        args.feat_dim = 128
    elif args.model == 'resnet18' or 'vit_small':
        args.feat_dim = 512    
    else:
        args.feat_dim = 2048

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
        batch_size = 1 if not k == 'train' and args.ood_method =='gradnorm' else args.batch_size
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

    if args.in_dataset == 'cifar-10-100':
        file_name = '{}_{}_{}_{}.csv'.format(args.in_dataset, args.out_num, args.model, args.loss)
        if 'CS' in args.loss_strategy:
            file_name = '{}_{}_{}_{}_cs.csv'.format(args.in_dataset, args.out_num, args.model, args.loss)
    else:
        file_name = '{}_{}_{}.csv'.format(args.in_dataset, args.model, args.loss)
        if 'CS' in args.loss_strategy:
            file_name = '{}_{}_{}_cs.csv'.format(args.in_dataset, args.model, args.loss)

    # TRAIN
    res = main_worker(options, args)

    # LOG
    res['split_idx'] = args.split_idx
    res['unknown'] = args.open_set_classes
    res['known'] = args.train_classes
    res['ID'] = args.log_dir.split("/")[-1]
    print(res)
    results[str(args.split_idx)] = res
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(dir_path, file_name), mode='a', header=False)
    print(args.out_dir)
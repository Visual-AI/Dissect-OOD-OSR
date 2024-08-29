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
import random
# from utils.tinyimages_80mn_loader import TinyImages

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.resnet import resnet50

from methods.ARPL.core import train_large, test
from methods.OOD.util.data_loader import get_loader_in, get_loader_out
from data.open_set_datasets import get_class_splits, get_datasets

from utils.utils import seed_torch, str2bool, load_networks, save_networks
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

parser = argparse.ArgumentParser("ARPLoss")

# dataset
parser.add_argument('--data-dir',type=str, default='/disk1/hjwang/datasets',help='dir of output')
parser.add_argument('--in-dataset', type=str, default='imagenet')
parser.add_argument('--out-datasets', default=['imagenet-r'], type=list, help="['imagenet-c', 'imagenet-r', 'cub', 'scars', 'aircraft'], ['SVHN', 'LSUN', 'LSUN_R', 'iSUN', 'dtd', 'places365'], ['inat', 'sun50', 'places50', 'dtd']")
parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")

# optimization
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")

# model
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--feat_dim', type=int, default=2048, help="Feature vector dim")

# aug
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--rand_aug_m', type=int, default=None)

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resume',type=str2bool, default=True, help='whether to resume training')
parser.add_argument('--resume-dir',type=str, default='/disk1/hjwang/osrd/logs/imagenet_resnet50_OE/bestpoint.pth.tar', help='dir of output')

parser.add_argument('--ood_method', default='', type=str, help='mls/odin/energy/react')
parser.add_argument('--loss_strategy', default='CE', type=str, help="CE/OE/ARPL/ARPL_CS")
parser.add_argument('--config_strategy', default='bas_config', type=str, help="bas_config/mls_config")
parser.add_argument('--ablation', type=str, default='.', help='ablation study')
parser.add_argument('--out-dir',type=str, default='./feat',help='dir of output')

parser.add_argument('--is_vis',type=str2bool, default=False, help='running ood/osr CKA')

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.manual_seed_all(args.seed)

args.ablation = '{}_{}_{}'.format(args.in_dataset, args.model, args.loss_strategy)
args.out_dir = os.path.join(args.out_dir, args.ablation)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if args.loss_strategy == 'OE':
    args.batch_size = 64
    args.oe_batch_size = args.batch_size*2

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]


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
    
def make_layers(model, pbar, options):
    model.eval()
    
    def hook(module, fea_in, fea_out):
        module_name.append(module.__class__)
        feat_hook.append(torch.sum(fea_out.clone().detach(), dim=0))
        return None

    count, num_convs = 0, 0
    ###################################Out-of-Distributions#####################################
    for batch_idx, tuples in enumerate(pbar):
        
        if len(tuples) == 2:
            data, labels = tuples
        elif len(tuples) == 3:
            data, labels, idx = tuples

        module_name = []
        feat_hook = []
        handles = []

        data = data.to(options['device'])
        model_chilren = model.modules()

        for m in model_chilren:
            count += data.size(0)
            if isinstance(m, (torch.nn.modules.conv.Conv2d)):
                handles.append(m.register_forward_hook(hook=hook))
                if batch_idx == 0:
                    num_convs += 1

        model(data, return_feat=False)

        for h in handles:
            h.remove()

        if batch_idx == 0:
            res = []
            for i in range(len(feat_hook)):
                res.append(torch.sum(feat_hook[i], dim=0))
        else:
            for i in range(len(res)):
                res[i] += torch.sum(feat_hook[i], dim=0)

        del feat_hook, module_name

    for i in range(len(res)):
        res[i] = res[i].data.cpu().numpy() / count

    return res, num_convs


# def make_layers(net, pbar, options):
#     layer1, layer2, layer3, layer4 = 0,0,0,0
#     total = 0

#     with torch.no_grad():
#         for batch_idx, tuples in enumerate(pbar):
#             if len(tuples) == 2:
#                 data, labels = tuples
#             elif len(tuples) == 3:
#                 data, labels, idx = tuples
#             data, labels = data.to(options['device']), labels.to(options['device'])
#             total += data.size(0)

#             with torch.no_grad():
#                 l1, l2, l3, l4 = net.feature_list(data)

#                 if batch_idx == 0:
#                     layer1, layer2, layer3, layer4 = torch.sum(l1, dim=0), torch.sum(l2, dim=0), torch.sum(l3, dim=0), torch.sum(l4, dim=0)
#                 else:
#                     layer1 += torch.sum(l1, dim=0)
#                     layer2 += torch.sum(l2, dim=0)
#                     layer3 += torch.sum(l3, dim=0) 
#                     layer4 += torch.sum(l4, dim=0)

#     layer1, layer2, layer3, layer4 = torch.sum(layer1, dim=0), torch.sum(layer2, dim=0), torch.sum(layer3, dim=0), torch.sum(layer4, dim=0)
#     layer1, layer2, layer3, layer4 = layer1 / (total), layer2 / (total), layer3 / (total), layer4 / (total)

#     return [layer1.data.cpu().numpy(), layer2.data.cpu().numpy(), layer3.data.cpu().numpy(), layer4.data.cpu().numpy()]

def visualize(net, inloader, outloader, dataset, options):
    net.eval()
    torch.cuda.empty_cache()

    in_pbar = tqdm(inloader)
    out_pbar = tqdm(outloader)
    
    processed1, num_convs = make_layers(net, in_pbar, options)
    processed2, _ = make_layers(net, out_pbar, options)
    processed = np.array(processed1 + processed2)

    fig = plt.figure(figsize=(16, 8))
    for i in range(len(processed)):
        a = fig.add_subplot(num_convs//8+1, 8, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title("layer_{}".format(str(i % 4 + 1)), fontsize=14)
    
    fig = plt.gcf()
    plt.savefig(os.path.join(args.out_dir, dataset+'.pdf'), format='pdf', bbox_inches='tight')


def main():
    oeloader = None

    net = resnet50(pretrained=True)
    net = net.to(args.device)

    loader_in_train_dict = get_loader_in(args, split=('train'))
    trainloader = loader_in_train_dict.train_loader
    loader_in_dict = get_loader_in(args, split=('val'))
    testloaderIn, args.num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes

    options = vars(args)

    if 'ARPL' in args.loss_strategy:
        args.loss = "ARPLoss"
    elif args.loss_strategy == 'OE':
        args.loss = "OELoss"    
    else:
        args.loss = "Softmax"    

    Loss = importlib.import_module('methods.ARPL.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)
    criterion = criterion.to(args.device)

    if args.resume:
        if 'ARPL' in args.loss_strategy:
            if os.path.exists(os.path.join(args.out_dir, 'bestpoint.pth.tar')):
                net, criterion = load_networks(net, args.out_dir, 'bestpoint.pth.tar', options, criterion=criterion)
            elif os.path.exists(args.resume_dir):
                net, criterion = load_networks(net, args.resume_dir, 'bestpoint.pth.tar', options, criterion=criterion)

        else:
            if os.path.exists(args.resume_dir):
                net.load_state_dict(torch.load(args.resume_dir))
            else:
                net.load_state_dict(torch.load(os.path.join(args.out_dir, 'bestpoint.pth.tar')))


    if options['ood_method'] == 'ours':
        options['mean'] = cal_training_stats(net, trainloader, options)
        print(options['mean'])


    for out_dataset in args.out_datasets:
        print(out_dataset)
        if out_dataset in ['imagenet-c', 'imagenet-r']:
            if not out_dataset == 'imagenet-c':
                testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader
                if args.is_vis:
                    visualize(net, testloaderIn, testloaderOut, out_dataset, options)
                else:
                    ans = test(net, criterion, testloaderIn, testloaderOut, epoch=0, **options)
                    print(ans)
                
            else:
                acc, auroc = 0, 0
                for distortion_name in distortions:
                    for severity in range(1, 6):
                        options['distortion_name'], options['severity'] = distortion_name, str(severity)
                        testloaderOut = get_loader_out(args, (None, out_dataset), split='val', options=options).val_ood_loader
                        ans = test(net, criterion, testloaderIn, testloaderOut, epoch=0, **options)
                        acc += ans['ACC']
                        auroc += ans['AUROC']
                
                print(acc / (len(distortions)*6), auroc / (len(distortions)*6))

        elif out_dataset in ['cub', 'scars', 'aircraft']:
            if not out_dataset == 'scars':
                args.train_classes, args.open_set_classes = get_class_splits(out_dataset, 0, cifar_plus_n=10)
            else:
                args.train_classes, args.open_set_classes = range(120), range(120, 196)

            datasets = get_datasets(out_dataset, transform='rand-augment', train_classes=args.train_classes,
                                    open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                    split_train_val=False, image_size=args.image_size, seed=args.seed,
                                    args=args)
            dataloaders = {}
            for k, v, in datasets.items():
                dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=16)
            testloaderOut = dataloaders['val']
            if args.is_vis:
                visualize(net, testloaderIn, testloaderOut, out_dataset, options)
            else:
                ans = test(net, criterion, testloaderIn, testloaderOut, epoch=0, **options)
            print(ans)



if __name__ == '__main__':
    main()
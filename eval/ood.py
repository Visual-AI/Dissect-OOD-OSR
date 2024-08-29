import os
import argparse
import os.path as osp
import numpy as np
import warnings
import importlib
warnings.filterwarnings('ignore')
from tqdm import tqdm
import torch
import timm 

from methods.ARPL.arpl_models.resnetABN import resnet18ABN
from methods import test
from methods.OOD.util.data_loader import get_loader_in, get_loader_out
from utils.utils import load_networks, strip_state_dict
from utils.cls_name import obtain_ImageNet_classes, obtain_cifar10_classes, obtain_cifar100_classes
from utils.clip_tools import CLIP_ft, CLIP_lp
from utils.stools import get_GMM_stat

parser = argparse.ArgumentParser("OOD_eval")

# dataset
parser.add_argument('--data-dir',type=str, default='/disk/datasets/ood_zoo', help='dir of output')
parser.add_argument('--val_comb', type=str)
parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")

# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--temp', type=float, default=1.0, help="temp")

# model
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim")
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--magnitude', type=float, help="parameter for odin")

# aug
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")
parser.add_argument('--transform', type=str, default='rand-augment')

parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--rand_aug_m', type=int, default=None)

# misc
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resume-dir',type=str, default='', help='dir of output')
parser.add_argument('--resume_godin-dir',type=str, default='', help='dir of output')
parser.add_argument('--resume_crit-dir',type=str, default='', help='dir of output')
parser.add_argument('--resume_godin_crit-dir',type=str, default='', help='dir of output')

parser.add_argument('--ood_method', default='', type=str, help='mls/odin/energy/react')
parser.add_argument('--loss_strategy', default='', type=str, help="CE/OE/ARPL/ARPL_CS")

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.val_comb == 'small':
    args.in_dataset = 'cifar-10'
    args.out_datasets = ['SVHN', 'dtd', 'LSUN', 'LSUN_R', 'iSUN', 'places365']
    args.num_classes = 10
elif args.val_comb == 'middle':
    args.in_dataset = 'cifar-100'
    args.out_datasets = ['SVHN', 'dtd', 'LSUN', 'LSUN_R', 'iSUN', 'places365']
    args.num_classes = 100
elif args.val_comb == 'large':
    args.in_dataset = 'imagenet'
    # args.out_datasets = ['imagenet-r', 'imagenet-c']
    args.out_datasets = ['SVHN', 'dtd', 'LSUN', 'LSUN_R', 'iSUN', 'places365']

    args.num_classes = 1000
else:
    assert False, 'Not supported dataset combination: {}'.format(args.val_comb)

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]


def get_model(num_classes):
    if args.model == 'resnet18':
        from models.resnet import resnet18_cifar
        model = resnet18_cifar(num_classes=num_classes)
    elif args.model == 'resnet50':
        if args.in_dataset == 'imagenet':
            from models.resnet import resnet50
            model = resnet50(pretrained=args.loss_strategy=='CE')
        else:
            from models.resnet import resnet50_cifar
            model = resnet50_cifar(num_classes=num_classes)            
    elif args.model == 'wrn':
        from models.wrn import WideResNet
        model = WideResNet(40, num_classes, 2, dropRate=0.3)
    elif args.model == 'densenet121':
        from models.densenet import densenet121
        model = densenet121(num_classes=num_classes)
    elif args.model == 'vit':
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    elif args.model == 'clip':
        from transformers import CLIPModel, CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        return tokenizer, model
    elif args.model == 'clip_ft':
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIP_ft()
        return tokenizer, model
    elif args.model == 'clip_lp':
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIP_lp()
        return tokenizer, model
    else:
        assert False, 'Not supported model arch: {}'.format(args.model)

    if args.ood_method == 'godin':
        from models.godin_net import UpperNet
        model = UpperNet(model, args.feat_dim, num_classes)

    return model


def main():
    args.scene = 'ood'

    if args.ood_method == 'godin':
        print(args.resume_godin_dir)
    else:
        print(args.resume_dir)

    if args.in_dataset == 'imagenet':
        trainloader = get_loader_in(args, split=('train')).train_loader
        loader_in_dict = get_loader_in(args, split=('val'))
        testloaderIn, args.num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    else:
        loader_in_dict = get_loader_in(args)
        trainloader, testloaderIn, args.num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes 

    tokenizer = None
    if 'CS' in args.loss_strategy:
        if args.model == 'resnet18':
            net = resnet18ABN(num_classes=args.num_classes, num_bns=2)
    else:
        if 'clip' in args.model:
            tokenizer, net = get_model(args.num_classes)
        else:
            net = get_model(args.num_classes)

    net = net.to(args.device)

    if 'ARPL' in args.loss_strategy:
        args.loss = "ARPLoss"
        if args.in_dataset == 'imagenet':
            args.feat_dim = 2048

    options = vars(args)
    if args.ood_method == 'sem':
        feature_type_list = ['stat', 'mean', 'mean', 'mean', 'flat']
        reduce_dim_list = ['pca_50', 'none', 'none', 'none', 'pca_50']
        num_clusters_list = [3, 1, 1, 1, 10]
        feature_mean, feature_prec, component_weight_list, transform_matrix_list = get_GMM_stat(net, trainloader, num_clusters_list, feature_type_list, reduce_dim_list, options)
        options.update(
            {
                'feature_type_list': feature_type_list,
                'feature_mean': feature_mean,
                'feature_prec': feature_prec,
                'component_weight_list': component_weight_list,
                'transform_matrix_list': transform_matrix_list
            }
        )

    Loss = importlib.import_module('methods.loss.'+args.loss)
    criterion = getattr(Loss, args.loss)(**options)
    criterion = criterion.to(args.device)

    if 'ARPL' in args.loss:
        if args.ood_method == 'godin':
            args.resume_dir, args.resume_crit_dir = args.resume_godin_dir, args.resume_godin_crit_dir
        net, criterion = load_networks(net, options=options, model_dir=args.resume_dir, crit_dir=args.resume_crit_dir, criterion=criterion)
    elif args.ood_method == 'godin':
        if os.path.isfile(args.resume_godin_dir):
            state_dict = strip_state_dict(torch.load(args.resume_godin_dir), 'base_model.')
            net.load_state_dict(state_dict)
        else:
            assert False, 'Not exist file.'
    else:
        if args.in_dataset == 'imagenet' or 'vit' in args.model or 'clip' in args.model:
            if args.model == 'clip_lp':
                state_dict = torch.load("/disk/work/hjwang/pretrained_models/clip/linear_probe/clip_lp.pt")
                net.load_state_dict(state_dict)
            elif args.model == 'clip_ft':
                state_dict = torch.load("/disk/work/hjwang/pretrained_models/clip/finetuned/clip_ft.pt")
                net.load_state_dict(state_dict)
            else:
                pass
        else:
            state_dict = strip_state_dict(torch.load(args.resume_dir))
            net.load_state_dict(state_dict)

    if 'clip' in args.model:
        if args.in_dataset == 'imagenet':
            test_labels = obtain_ImageNet_classes() 
        elif args.in_dataset == 'cifar-10':
            test_labels = obtain_cifar10_classes()
    else:
        test_labels = None

    for out_dataset in args.out_datasets:
        if not out_dataset == 'imagenet-c':
            testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader
            res = test(net, criterion, testloaderIn, testloaderOut, tokenizer, test_labels, **options)
            res['Loss'] = args.loss_strategy
            res['Ood_method'] = args.ood_method
            res['Dataset'] = args.in_dataset + '-' + out_dataset
            print(res)

        else:
            acc, auroc = 0, 0
            for distortion_name in distortions:
                for severity in range(1, 6):
                    options['distortion_name'], options['severity'] = distortion_name, str(severity)
                    testloaderOut = get_loader_out(args, (None, out_dataset), split='val', options=options).val_ood_loader
                    res = test(net, criterion, testloaderIn, testloaderOut, tokenizer, test_labels, **options)
                    acc += res['ACC']
                    auroc += res['AUROC']

            res['ACC'] = acc / (len(distortions)*6)
            res['AUROC'] = auroc / (len(distortions)*6)
            res['Loss'] = args.loss_strategy
            res['Ood_method'] = args.ood_method
            res['Dataset'] = args.in_dataset + '-' + out_dataset
            print(res)


if __name__ == '__main__':
    main()

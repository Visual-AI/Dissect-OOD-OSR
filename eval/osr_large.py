import os
import argparse
import datetime
import time
import pandas as pd
import importlib
import pickle

import torch
from torch.utils.data import DataLoader

from methods import test

from utils.utils import seed_torch, str2bool, strip_state_dict, load_networks
from utils.cls_name import obtain_ImageNet_classes, obtain_cifar10_classes, obtain_cifar100_classes

from utils.stools import get_GMM_stat

from data.open_set_datasets import get_datasets
from utils.clip_tools import CLIP_ft, CLIP_lp

from models.model_utils import get_model

from config import osr_split_dir

parser = argparse.ArgumentParser("OSR_eval")

# Dataset
parser.add_argument('--in-dataset', type=str, default='cub', help="")
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=448)

# optimization
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing. No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str)
parser.add_argument('--feat_dim', type=int, default=None, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

parser.add_argument('--train_feat_extractor', default=True, type=str2bool, help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--is_pretrained', default=False, type=str2bool, help='Do we use ImageNet pretrained model', metavar='BOOL')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco', help='Which pretraining to use if --model=timm_resnet50_pretrained. Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')

# misc
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--resume-dir',type=str, default='', help='dir of output')
parser.add_argument('--resume_crit-dir',type=str, default='', help='dir of output')

parser.add_argument('--ood_method', default='mls', type=str, help="mls/odin/energy/react")
parser.add_argument('--loss_strategy', default='', type=str, help="CE/OE/ARPL/ARPL_CS")

args = parser.parse_args()


if __name__ == '__main__':
    print(args.resume_dir)

    args.scene = 'osr'
    
    # args = get_default_hyperparameters(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = dict()

    # SEED
    seed_torch(args.seed)
    # INIT
    args.feat_dim = 2048
    # SAVE PARAMS
    options = vars(args)

    # Get OSR splits
    osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.in_dataset))

    with open(osr_path, 'rb') as f:
        class_info = pickle.load(f)

    if args.in_dataset == 'imagenet21k':
        train_classes = class_info['closest_easy_i1k_classes'] + class_info['closest_hard_i1k_classes']
    else:
        train_classes = class_info['known_classes']
        open_set_classes = class_info['unknown_classes']

    for difficulty in ('Easy', 'Hard'):

        # ------------------------
        # DATASETS
        # ------------------------
        args.train_classes = train_classes
        if args.in_dataset == 'imagenet21k':
            args.open_set_classes = None # placeholder
        else:
            args.open_set_classes = open_set_classes[difficulty]

        if difficulty == 'Hard' and (args.in_dataset not in ['imagenet', 'imagenet21k']):
            args.open_set_classes += open_set_classes['Medium']

        datasets = get_datasets(args.in_dataset, transform=args.transform, train_classes=args.train_classes,
                                image_size=args.image_size, balance_open_set_eval=False,
                                split_train_val=False, open_set_classes=args.open_set_classes,
                                osr_split=difficulty if args.in_dataset == 'imagenet21k' else None)
        # ------------------------
        # DATALOADERS
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=8)

        options.update(
            {
                'known':    args.train_classes,
                'unknown':  args.open_set_classes,
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
        # outloader = get_loader_out(args, (None, 'waterbird'), split='val').val_ood_loader
        
        if args.model == 'clip':
            from transformers import CLIPModel, CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        elif args.model == 'clip_ft':
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            model = CLIP_ft()
        elif args.model == 'clip_lp':
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            model = CLIP_lp()
        else:
            tokenizer = None
            model = get_model(args, wrapper_class=None, evaluate=True)
        
        if args.loss == 'ARPLoss':
            model, criterion = load_networks(model, options=options, model_dir=args.resume_dir, crit_dir=args.resume_crit_dir, criterion=criterion)
        else:
            if 'vit' in args.model or 'clip' in args.model:
                if args.model == 'clip_lp':
                    state_dict = torch.load("/disk/work/hjwang/pretrained_models/clip/linear_probe/clip_lp.pt")
                    model.load_state_dict(state_dict)
                elif args.model == 'clip_ft':
                    state_dict = torch.load("/disk/work/hjwang/pretrained_models/clip/finetuned/clip_ft.pt")
                    model.load_state_dict(state_dict)
                else:
                    pass
            else:
                filename = strip_state_dict(torch.load(args.resume_dir))
                model.load_state_dict(filename)

        model = model.to(args.device)
        
        if args.ood_method == 'sem':
            feature_type_list = ['stat', 'mean', 'mean', 'mean', 'flat']
            reduce_dim_list = ['pca_50', 'none', 'none', 'none', 'pca_50']
            num_clusters_list = [3, 1, 1, 1, 10]
            feature_mean, feature_prec, component_weight_list, transform_matrix_list = get_GMM_stat(model, trainloader, num_clusters_list, feature_type_list, reduce_dim_list, options)
            options.update(
                {
                    'feature_type_list': feature_type_list,
                    'feature_mean': feature_mean,
                    'feature_prec': feature_prec,
                    'component_weight_list': component_weight_list,
                    'transform_matrix_list': transform_matrix_list
                }
            )

        if 'clip' in args.model:
            test_labels = obtain_ImageNet_classes()
        else:
            test_labels = None

        res = test(model, criterion, testloader, outloader, tokenizer, test_labels, **options)
        res['Loss'] = args.loss_strategy
        res['Ood_method'] = args.ood_method
        res['Dataset'] = args.in_dataset + '-' + difficulty

        print(res)
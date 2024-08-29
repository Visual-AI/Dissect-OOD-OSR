# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm

import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from utils.utils import seed_torch, str2bool

from data.open_set_datasets import get_class_splits, get_datasets
from methods.OOD.util.data_loader import get_loader_in, get_loader_out
from utils.tinyimages_80mn_loader import TinyImages

import sys
from os import path

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier with OE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir',type=str, default='/disk/datasets/ood_zoo',help='dir of output')
parser.add_argument('--in-dataset', type=str, default='cifar10', help='Choose between cifar-10-10, cifar-10-100-10, cifar-10-100-50, tinyimagenet')
parser.add_argument('--model', '-m', type=str, default='resnet50', help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.000003, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')

parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--rand_aug_m', type=int, default=None)

parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--loss_strategy', default='OE', type=str, help="CE/OE/ARPL/ARPL_CS")

# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/oe_scratch', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--ablation', type=str, help='conv-default/oe-default')

args = parser.parse_args()

# args.image_size = 64 if args.in_dataset == 'tinyimagenet' else 32
# mean = [x / 255 for x in [125.3, 123.0, 113.9]] if not args.in_dataset == 'tinyimagenet' else [0.485, 0.456, 0.406]
# std = [x / 255 for x in [63.0, 62.1, 66.7]] if not args.in_dataset == 'tinyimagenet' else [0.229, 0.224, 0.225]

args.image_size = 224

transform_train_largescale = trn.Compose([
    trn.Resize(256),
    trn.RandomHorizontalFlip(),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if args.in_dataset in ['cifar-10', 'cifar-100', 'imagenet']:
    args.is_ood = True
else:
    args.is_ood = False

if 'cifar-10-100' in args.in_dataset:
    args.out_num = int(args.in_dataset.split('-')[-1])
    args.in_dataset = 'cifar-10-100'

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)


trainloader = get_loader_in(args, split=('train')).train_loader
loader_in_dict = get_loader_in(args, split=('val'))
testloader, args.num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes


# ood_data = TinyImages(os.path.join(args.data_dir, '300K_random_images.npy'), transform=trn.Compose(
#     [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
#      trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))

# oeloader = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

# oeloader = torch.utils.data.DataLoader(
#                 torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'imagenet-r'), transform_train_largescale),
#                 batch_size=args.batch_size, shuffle=True)
# valset = torchvision.datasets.SUN397(root=os.path.join('./', 'data'), download=True, transform=transform_train_largescale)
# oeloader = torch.utils.data.DataLoader(valset, batch_size=args.oe_batch_size, shuffle=True)

class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx
    
valset = ImageNetBase(root=os.path.join('/disk/datasets/imagenet21k_resized', 'imagenet21k_train'), transform=transform_train_largescale)
oeloader = torch.utils.data.DataLoader(valset, batch_size=args.oe_batch_size, shuffle=True)

# Create model
# net = resnet50(weights=None)
net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

start_epoch = 0
# Restore model if desired
# if args.load != '':
#     for i in range(1000 - 1, -1, -1):
#         model_name = os.path.join(args.load, args.in_dataset + '_' + args.model + '_oe_scratch_epoch_' + str(i) + '.pt')
#         if os.path.isfile(model_name):
#             net.load_state_dict(torch.load(model_name))
#             print('Model restored! Epoch:', i)
#             start_epoch = i + 1
#             break
#     if start_epoch == 0:
#         assert False, "could not resume"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(trainloader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    oeloader.dataset.offset = np.random.randint(len(oeloader.dataset))
    pbar = tqdm(zip(trainloader, oeloader), total=len(trainloader))

    for in_set, out_set in pbar:
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(testloader)
    state['test_accuracy'] = correct / len(testloader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.in_dataset + '_' + args.model + '_oe_scratch_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(), os.path.join(args.save, args.in_dataset + '_' + args.model + '_oe_scratch_epoch_' + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.in_dataset + '_' + args.model + '_oe_scratch_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): 
        os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.in_dataset + '_' + args.model + '_oe_scratch_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Acc {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100. * state['test_accuracy'])
    )

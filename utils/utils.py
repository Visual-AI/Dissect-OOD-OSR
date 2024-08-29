import os
import torch
import random
import numpy as np
import os.path as osp
import os


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def save_networks(networks, out_dir, name, options, criterion=None):
    weights = networks.state_dict()
    filename = os.path.join(out_dir, name)
    if options['model'] == 'vit_small':
        state = {"model": weights}
        torch.save(state, filename)
    else:
        torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        torch.save(weights, os.path.join(out_dir, 'criterion.pth.tar'))


def load_networks(networks, out_dir=None, name=None, options=None, model_dir=None, crit_dir=None, criterion=None):
    weights = networks.state_dict()
    if model_dir is None:
        filename = os.path.join(out_dir, name)
    else:
        filename = model_dir

    if options['model'] == 'vit_small':
        networks = torch.nn.DataParallel(networks)
        networks.load_state_dict(torch.load(filename)['model'], map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(filename, map_location=torch.device('cpu'))
        if options['in_dataset'] == 'imagenet':
            new_state_dict = strip_state_dict(state_dict, 'module.resnet.')
        else:
            new_state_dict = strip_state_dict(state_dict, 'module.')
        # new_state_dict = state_dict
        networks.load_state_dict(new_state_dict)
    if criterion:
        weights = criterion.state_dict()
        if crit_dir is None:
            critname = os.path.join(out_dir, 'criterion.pth.tar')
        else:
            critname = crit_dir
        criterion.load_state_dict(torch.load(critname, map_location=torch.device('cpu')))

    return networks, criterion


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("Training")

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar-10-10', help="")
    parser.add_argument('--loss', type=str, default='ARPLoss', help='For cifar-10-100')

    args = parser.parse_args()

    for dataset in ('mnist', 'svhn', 'cifar-10-10', 'cifar-10-100', 'tinyimagenet'):
        args.dataset = dataset
        args = get_default_hyperparameters(args)
        print(f'{dataset}')
        print(args)
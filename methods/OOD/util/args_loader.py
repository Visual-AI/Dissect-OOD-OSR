import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='CIFAR-100 imagenet')
    parser.add_argument('--out-datasets', default=['SVHN', 'dtd', 'LSUN', 'LSUN_R', 'iSUN', 'places365'], type=list, help="['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")
    parser.add_argument('--name', default="resnet18", type=str, help='neural network name and training set')
    parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture [resnet50]')
    parser.add_argument('--threshold', default=1.0, type=float, help='sparsity level')
    parser.add_argument('--method', default='energy', type=str, help='odin mahalanobis CE_with_Logst')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
    parser.add_argument('--data-dir',type=str, default='/home/hjwang/osrd/methods/OOD/datasets',help='dir of output')

    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('-b', '--batch-size', default=25, type=int, help='mini-batch size')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.set_defaults(argument=True)
    args = parser.parse_args()
    return args
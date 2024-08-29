import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from torchvision.datasets import SVHN
from data.augmentations.randaugment import RandAugment
from utils.tinyimages_80mn_loader import TinyImages


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_clip = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),  # for CLIP
    ])

transform_vis = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_vis_largescale = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_vis_largescale_grey = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class cat_dataloaders():
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise StopIteration
        return tuple(out)

    def __len__(self):
        return min(len(data_loader) for data_loader in self.dataloaders)


def get_loader_in(args, split=('train', 'val')):
    global transform_clip
    global transform_train, transform_test, transform_train_largescale, transform_test_largescale, transform_vis_largescale

    if 'clip' in args.model:
        transform_train = transform_clip
        transform_test = transform_clip
        transform_train_largescale = transform_clip
        transform_test_largescale = transform_clip
        transform_vis_largescale = transform_clip
    else:
        transform_train = transform_train
        transform_test = transform_test
        transform_train_largescale = transform_train_largescale
        transform_test_largescale = transform_test_largescale
        transform_vis_largescale = transform_vis_largescale
        
    train_loader, val_loader, lr_schedule, num_classes = None, None, [50, 75, 90], 0
    if 'no' in args.transform:
        shuffle = False
    else:
        shuffle = True

    if args.transform == 'rand-augment':
        if args.rand_aug_m is not None:
            if args.rand_aug_n is not None:
                transform_train.transforms.insert(0, RandAugment(args.rand_aug_m, args.rand_aug_n, args=args))
                transform_train_largescale.transforms.insert(0, RandAugment(args.rand_aug_m, args.rand_aug_n, args=args))

    if args.in_dataset in ["cifar10", "cifar-10", "CIFAR-10"]:
        # Data loading code
        if 'train' in split:
            if 'no' in args.transform:
                transform_train = transform_vis
            trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'id_data'), train=True, download=True, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'id_data'), train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
        num_classes = 10

    elif args.in_dataset in ["cifar100", "cifar-100", "CIFAR-100"]:
        # Data loading code
        if 'train' in split:
            if 'no' in args.transform:
                transform_train = transform_vis
            trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'id_data'), train=True, download=True, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'id_data'), train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
        num_classes = 100

    elif args.in_dataset == "imagenet":
        # Data loading code
        if 'train' in split:
            if 'no' in args.transform:
                train_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'id_data', 'imagenet', 'train'), transform_vis_largescale),
                    batch_size=args.batch_size, shuffle=shuffle, **kwargs)
            elif 'imagenet_r' in args.transform:
                from utils.ImagenetR_ImageFolder import ImageFolder
                train_loader = torch.utils.data.DataLoader(
                    ImageFolder(os.path.join(args.data_dir, 'id_data', 'imagenet', 'train'), transform_vis_largescale),
                    batch_size=args.batch_size, shuffle=shuffle, **kwargs)                
            else:
                from utils.ImageNetFolder import ImageFolder
                train_loader = torch.utils.data.DataLoader(
                    ImageFolder(os.path.join(args.data_dir, 'id_data', 'imagenet', 'train'), transform_train_largescale),
                    batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'id_data', 'imagenet', 'val'), transform_test_largescale),
                batch_size=args.batch_size, shuffle=False, **kwargs)
        num_classes = 1000


    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes,
    })


def get_loader_out(args, dataset=(''), split=('train', 'val'), options=None):
    global transform_train, transform_test, transform_train_largescale, transform_test_largescale

    train_ood_loader, val_ood_loader, num_classes = None, None, 0

    if 'train' in split:
        if dataset[0].lower() == 'imagenet':
            train_ood_loader = torch.utils.data.DataLoader(
                ImageNet(transform=transform_train),
                batch_size=args.batch_size, shuffle=True, **kwargs)
        elif dataset[0].lower() == 'tim':
            train_ood_loader = torch.utils.data.DataLoader(
                TinyImages(transform=transform_train),
                batch_size=args.batch_size, shuffle=True, **kwargs)

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch_size

        transform = transform_test_largescale if args.transform == 'imagenet_r' else transform_test

        if val_dataset == 'SVHN':
            val_ood_loader = torch.utils.data.DataLoader(SVHN(os.path.join(args.data_dir, 'ood_data', 'svhn'), split='test', transform=transform, download=True), batch_size=batch_size, shuffle=False, num_workers=2)
            num_classes = 10
        elif val_dataset == 'dtd':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir, 'ood_data', 'dtd/images'), transform=transform),
                                                       batch_size=batch_size, shuffle=False, num_workers=2)
            num_classes = 47
        elif val_dataset == 'LSUN':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'LSUN'),
                                                          transform=transform), batch_size=batch_size, shuffle=False, num_workers=2)
            num_classes = 10
        elif val_dataset == 'iSUN':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'iSUN'),
                                                          transform=transform), batch_size=batch_size, shuffle=False, num_workers=2)
            num_classes = 10
        elif val_dataset == 'LSUN_R':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'LSUN_resize'),
                                                          transform=transform), batch_size=batch_size, shuffle=False, num_workers=2)
            num_classes = 10
        elif val_dataset == 'places365':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'Places'),
                                                          transform=transform), batch_size=batch_size, shuffle=False, num_workers=2)       
            num_classes = 365
        elif val_dataset == 'cifar-100':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'id_data'), train=False, download=True, transform=transform),
                                                       batch_size=batch_size, shuffle=False, num_workers=2)
            num_classes = 100
        elif val_dataset == 'imagenet':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'imagenet', 'val'), transform_test_largescale),
                batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 1000
        elif val_dataset == 'imagenet-c':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'imagenet-c', options['distortion_name'], options['severity']), transform_test_largescale),
                batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 1000
        elif val_dataset == 'imagenet-r':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'imagenet-r'), transform_test_largescale),
                batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 1000
        elif val_dataset == 'cifar-10':
            valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'id_data'), train=False, download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 10
        elif val_dataset == 'cifar-100':
            valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'id_data'), train=False, download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 100
        elif val_dataset == 'MNIST':
            valset = torchvision.datasets.MNIST(root=os.path.join('./', 'data'), train=False, download=True, transform=transform_vis_largescale_grey)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 10
        elif val_dataset == 'EMNIST':
            valset = torchvision.datasets.EMNIST(root=os.path.join('./', 'data'), split='letters', train=False, download=True, transform=transform_vis_largescale_grey)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 10
        elif val_dataset == 'QMNIST':
            valset = torchvision.datasets.QMNIST(root=os.path.join('./', 'data'), train=False, download=True, transform=transform_vis_largescale_grey)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 10
        elif val_dataset == 'FashionMNIST':
            valset = torchvision.datasets.FashionMNIST(root=os.path.join('./', 'data'), train=False, download=True, transform=transform_vis_largescale_grey)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 10
        elif val_dataset == 'Food101':
            valset = torchvision.datasets.Food101(root=os.path.join('./', 'data'), download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 101
        elif val_dataset == 'STL10':
            valset = torchvision.datasets.STL10(root=os.path.join('./', 'data'), download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 10
        elif val_dataset == 'OxfordIIITPet':
            valset = torchvision.datasets.OxfordIIITPet(root=os.path.join('./', 'data'), download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 397
        elif val_dataset == 'Flowers102':
            valset = torchvision.datasets.Flowers102(root=os.path.join('./', 'data'), download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 102
        elif val_dataset == 'SUN397':
            valset = torchvision.datasets.SUN397(root=os.path.join('./', 'data'), download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 397
        elif val_dataset == 'iNaturalist':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'ood_data', 'iNaturalist'),
                                                 transform=transform_test_largescale), batch_size=batch_size, shuffle=False, num_workers=2)
        elif val_dataset == 'Caltech101':
            valset = torchvision.datasets.Caltech101(root=os.path.join('./', 'data'), download=True, transform=transform_vis_largescale_grey)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 101
        elif val_dataset == 'Country211':
            valset = torchvision.datasets.Country211(root=os.path.join('./', 'data'), download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 211
        elif val_dataset == 'EuroSAT':
            valset = torchvision.datasets.EuroSAT(root=os.path.join('./', 'data'), download=True, transform=transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)
            num_classes = 10

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader": val_ood_loader,
        "num_classes": num_classes,
    })
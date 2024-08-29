# ----------------------
# PROJECT ROOT DIR
# ----------------------
# project_root_dir = './'
# hyperparameter_dir = "/home/hjwang/osat/utils/paper_hyperparameters.csv"

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
# exp_root = './exp'        # directory to store experiment output (checkpoints, logs, etc)
# save_dir = './logs'    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
# root_model_path = './logs/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
# root_criterion_path = './logs/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
cifar_10_root = '/disk/datasets'                                    # CIFAR10
cifar_100_root = '/disk/datasets'                                  # CIFAR100
mnist_root = '/disk/datasets'                                               # MNIST
svhn_root = '/disk/datasets'                                                 # SVHN
tin_train_root_dir = '/disk/datasets/tiny-imagenet-200/train'        # TinyImageNet Train
tin_val_root_dir = '/disk/datasets/tiny-imagenet-200/val/images'     # TinyImageNet Val

cub_root = '/disk/datasets/ood_zoo/ood_data/CUB'                                                   # CUB
waterbird_root = '/disk/work/hjwang/osrd/waterbird'                                                # Waterbird
aircraft_root = '/disk/datasets/ood_zoo/ood_data/aircraft/fgvc-aircraft-2013b'                     # FGVC-Aircraft
car_root = "/disk/datasets/ood_zoo/ood_data/stanford_car/cars_{}/"                                 # Stanford Cars
meta_default_path = "/disk/datasets/ood_zoo/ood_data/stanford_car/devkit/cars_{}.mat"              # Stanford Cars Devkit
pku_air_root = '/work/sagar/data/pku-air-300/AIR'                                   # PKU-AIRCRAFT-300
imagenet_root = '/disk/datasets/ood_zoo/id_data/imagenet'                                   # ImageNet-1K
imagenet21k_root = '/disk/datasets/imagenet21k_resized'                       # ImageNet-21K-P

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir = '/disk/work/hjwang/osrd/data/open_set_splits'

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
imagenet_moco_path = '/disk/work/hjwang/osrd/SSB_models/cub/cross_entropy/cub_599_Softmax.pth'
places_moco_path = '/disk/work/hjwang/osrd/SSB_models/cub/cross_entropy/cub_599_Softmax.pth'
places_supervised_path = '/disk/work/hjwang/osrd/SSB_models/cub/cross_entropy/cub_599_Softmax.pth'
imagenet_supervised_path = '/disk/work/hjwang/osrd/SSB_models/cub/cross_entropy/cub_599_Softmax.pth'
clip_text_path = '/disk/work/hjwang/osrd/imagenet_class_clean.npy'
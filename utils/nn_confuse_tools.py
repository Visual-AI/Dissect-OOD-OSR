import os
import sys
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms

from methods.OOD.util.data_loader import get_loader_in, get_loader_out
from utils.tinyimages_80mn_loader import TinyImages

from utils.utils import seed_torch, str2bool, load_networks, strip_state_dict

# from utils.score import get_score
from utils.compute_threshold import get_threshold
from sklearn.metrics import roc_curve, roc_auc_score


def find_nearest(array, value):
    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx


def compute_t(open_preds, open_labels):
    fpr, tpr, thresh = roc_curve(open_labels, open_preds, drop_intermediate=False)
    _, idx = find_nearest(tpr, 0.95)
    t = thresh[idx]
    return t


def compute_auroc(open_set_preds, open_set_labels):
    auroc = roc_auc_score(open_set_labels, open_set_preds)
    return auroc


def find_pos_samples(mixed_preds_arr, images_arr, t, in_length, options):
    P_index = mixed_preds_arr > t
    P_images = images_arr[P_index].to(options['device'])
    return P_images, P_index


@torch.no_grad()
def extract_features(model, P_images, options):
    model.eval()

    print("Extracting features for positive samples...")
    num_images, imgs_per_chunk = P_images.shape[0], 256
    num_chunks = num_images // imgs_per_chunk

    for idx in range(0, num_images, imgs_per_chunk):
        Pimgs_in_chunk = P_images[idx : min((idx + imgs_per_chunk), num_images), :]
    
        if 'dino_vit' in options['model']:
            features_list = model.get_intermediate_layers(Pimgs_in_chunk)
            P_features_in_chunk = features_list[-1].view(Pimgs_in_chunk.size(0), -1)
        else:
            P_features_in_chunk, _ = model(Pimgs_in_chunk, True)
        
        P_features = P_features_in_chunk.cpu() if idx == 0 else torch.cat((P_features, P_features_in_chunk.cpu()), 0)

    P_features = nn.functional.normalize(P_features, dim=1, p=2)

    return P_features


@torch.no_grad()
def knn_classifier(model, pos_features, mixed_preds_arr, mixed_labels_arr, P_index, t, topk, options):
    trainloader, oeloader =  options['trainloader'], options['oeloader']

    pos_features = pos_features.to(options['device'])
    dist_list = []

    others_length = pos_features.size(0)
    changed_preds_arr = mixed_preds_arr[P_index]

    for loader_idx, loader in enumerate((oeloader, trainloader)):
        for batch_idx, batch in enumerate(tqdm(loader)):
            if len(batch) == 2:
                data, labels = [x.to(options['device']) for x in batch]
            elif len(batch) == 3:
                data, labels, idxs = [x.to(options['device']) for x in batch]

            train_features, _ = model(data, True)
            train_features = nn.functional.normalize(train_features, dim=1, p=2)
            train_features = train_features.t()
            similarity = torch.mm(pos_features, train_features)

            for j in range(others_length):
                distances, indices = similarity[j].topk(topk, largest=True, sorted=True)
                distances = distances.data.cpu().numpy().tolist()

                if batch_idx + loader_idx == 0:
                    for k in range(topk):
                        dist_list.append(distances[k])
                else:
                    tmp_dist_list = dist_list[j*topk:+(j+1)*topk]

                    for d in range(len(distances)):
                        flag = False
                        for dd in range(len(tmp_dist_list)):
                            if distances[d] > tmp_dist_list[dd]:
                                if not loader_idx == 0: 
                                    flag = True
                                    changed_preds_arr[j] = -1000
                                else:
                                    tmp_dist_list.insert(dd, distances[d])
                                break
                        if flag:
                            break
                    dist_list[j*topk:+(j+1)*topk] = tmp_dist_list[:topk]

    mixed_preds_arr[P_index] = changed_preds_arr
    return mixed_preds_arr


def test(net, criterion, inloader, oodloader, options):
    net.eval()

    preds_arr = {0: [], 1: []}
    labels_arr = {0: [], 1: []}

    if 'react' in options['ood_method']:
        if options['in_dataset'] in ['imagenet', 'cub', 'scars', 'aircraft', 'imagenet-r', 'imagenet-c']:
            options['threshold'] = 1.0
        else:
            options['threshold'] = get_threshold(net, inloader, options)
    else:
        options['threshold'] = 1e10
    
    forward_threshold = forward_fun(options)

    # For data loader
    for open_set_label, loader in enumerate((inloader, oodloader)):
        for batch_idx, batch in enumerate(tqdm(loader)):
            if len(batch) == 2:
                data, labels = [x.to(options['device']) for x in batch]
            elif len(batch) == 3:
                data, labels, idxs = [x.to(options['device']) for x in batch]

            with torch.no_grad():
                if 'dino_vit' in options['model']:
                    logits = net(data)
                else:
                    x, y = forward_threshold(data, net)
                    logits, _ = criterion(x, y)
                scores = -get_score(data, net, forward_threshold, criterion, options, logits)

            preds_arr[open_set_label].extend(scores)
            labels_arr[open_set_label].extend([open_set_label] * len(scores))

    in_length = len(preds_arr[0])
    preds_arr = np.array(preds_arr[0] + preds_arr[1])
    labels_arr = np.array(labels_arr[0] + labels_arr[1])

    return preds_arr, labels_arr, in_length


if __name__ == '__main__':
    mixed_preds_arr, mixed_labels_arr, imgs_arr, in_length = test(net, criterion, testloaderIn, testloaderOut, options)
    t = compute_t(mixed_preds_arr, mixed_labels_arr)
    pos_imgs, P_index = find_pos_samples(mixed_preds_arr, imgs_arr, t, in_length)
    pos_features = extract_features(net, pos_imgs, args)
    new_mixed_preds_arr = knn_classifier(net, trainloader, oeloader, pos_features, mixed_preds_arr, mixed_labels_arr, P_index, t)
import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from utils import evaluation
from utils.score import get_score
from utils.compute_threshold import get_threshold

from sklearn.metrics import accuracy_score


def forward_fun(options):
    def forward_threshold(inputs, model):
        if 'dino_vit' in options['model']:
            train_features_list = model.base_model.get_intermediate_layers(inputs, 4)
            feat = torch.cat([x[:, 0] for x in train_features_list], dim=-1)
            logits = model(inputs)
        else:
            feat, logits = model.forward_threshold(inputs, threshold=options['threshold'], using_ash='ash' in options['ood_method'], return_feat=True)
        return feat, logits
    return forward_threshold


def test_id(net, criterion, inloader, **options):
    net.eval()
    correct, total = 0, 0
    torch.cuda.empty_cache()

    id_pbar = tqdm(inloader)
    
    with torch.no_grad():
        for batch_idx, tuples in enumerate(id_pbar):
            if len(tuples) == 2:
                data, labels = tuples
            elif len(tuples) == 3:
                data, labels, idx = tuples
            data, labels = data.to(options['device']), labels.to(options['device'])
            
            with torch.no_grad():
                x, y = net(data, return_feat=True)
                logits, _ = criterion(x, y)
                total += labels.size(0)
                correct += (logits.data.max(1)[1] == labels.data).sum()

        id_acc = float(correct) * 100. / float(total)

    return id_acc


def test(net, criterion, inloader, outloader, tokenizer=None, test_labels=None, **options):
    net.eval()

    id_preds_arr = {0: [], 1: []}
    ood_preds_arr = {0: [], 1: []}

    id_labels_arr = {0: [], 1: []}
    ood_labels_arr = {0: [], 1: []}

    if 'react' in options['ood_method']:
        if options['in_dataset'] in ['imagenet', 'cub', 'scars', 'aircraft', 'imagenet-r', 'imagenet-c']:
            options['threshold'] = 1.0
        else:
            options['threshold'] = get_threshold(net, inloader, options)
    else:
        options['threshold'] = 1e10

    forward_threshold = forward_fun(options)

    for open_set_label, loader in enumerate((inloader, outloader)):
        for _, batch in enumerate(tqdm(loader)):
            if len(batch) == 2:
                data, labels = [x.to(options['device']) for x in batch]
            elif len(batch) == 3:
                data, labels, idxs = [x.to(options['device']) for x in batch]
            
            with torch.no_grad():
                if options['model'] == 'vit':
                    outputs = net(data)
                    logits = outputs.logits
                elif 'clip' in options['model']:
                    if options['model'] == 'clip_lp':
                        logits = net(data, labels)
                    elif options['model'] == 'clip_ft':
                        if options['scene'] == 'ood':
                            if options['in_dataset'] == 'imagenet':
                                text_inputs = tokenizer([f"a good photo of real {c}" for c in test_labels], padding=True, return_tensors="pt")
                            else:
                                name_idx = labels.detach().cpu().numpy()
                                name_idx[name_idx >= len(test_labels)] = len(test_labels) - 1
                                text_inputs = tokenizer([f"a good photo of real {test_labels[idx]}" for idx in name_idx], padding=True, return_tensors="pt")
                        elif options['scene'] == 'osr':
                            text_inputs = tokenizer([f"a good photo of real {c}" for c in test_labels], padding=True, return_tensors="pt")
                        logits, _ = net(data, text_inputs)
                    else:
                        image_features = net.get_image_features(pixel_values=data).float()
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        if options['scene'] == 'ood':
                            if options['in_dataset'] == 'imagenet':
                                text_inputs = tokenizer([f"a good photo of real {c}" for c in test_labels], padding=True, return_tensors="pt")
                            else:
                                name_idx = labels.detach().cpu().numpy()
                                name_idx[name_idx >= len(test_labels)] = len(test_labels) - 1
                                text_inputs = tokenizer([f"a good photo of real {test_labels[idx]}" for idx in name_idx], padding=True, return_tensors="pt")

                        elif options['scene'] == 'osr':
                            text_inputs = tokenizer([f"a good photo of real {c}" for c in test_labels], padding=True, return_tensors="pt")
                        text_features = net.get_text_features(input_ids = text_inputs['input_ids'].to(options['device']), 
                                                            attention_mask = text_inputs['attention_mask'].to(options['device'])).float()
                        text_features /= text_features.norm(dim=-1, keepdim=True)   
                        logits = image_features @ text_features.T
                else:
                    x, y = forward_threshold(data, net)
                    logits, _ = criterion(x, y)
                    
            scores = -get_score(data, net, forward_threshold, criterion, options, logits)

            # Update preds and labels
            id_preds_arr[open_set_label].extend(logits.data.cpu().numpy())
            id_labels_arr[open_set_label].extend(labels.cpu().numpy().tolist())

            ood_preds_arr[open_set_label].extend(scores)
            ood_labels_arr[open_set_label].extend([open_set_label] * len(scores))

    in_length = len(ood_preds_arr[0])
    ood_preds_arr = np.array(ood_preds_arr[0] + ood_preds_arr[1])
    ood_labels_arr = np.array(ood_labels_arr[0] + ood_labels_arr[1])

    results = evaluation.metric(id_preds_arr, id_labels_arr, ood_preds_arr, ood_labels_arr)
    # ACC = accuracy_score(np.array(id_labels_arr[0]), np.array(id_preds_arr[0]).argmax(axis=-1))
    # results['ACC'] = round(ACC, 4)

    return results
from __future__ import print_function
import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
import time

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_threshold(model, loader, options):
    activation_log = []

    model.eval()

    count = 0
    lim = 2000

    for batch_idx, tuples in enumerate(loader):
        if count > lim:
            break
            
        if len(tuples) == 2:
            data, labels = tuples
        elif len(tuples) == 3:
            data, labels, idx = tuples

        data = data.to(options['device'])

        curr_batch_size = data.shape[0]

        # inputs = data.float()

        with torch.no_grad():
            hooker_handles = []
            if options['model'] == 'vit_small':
                layer_remark = 'to_latent'
                hooker_handles.append(model.module.to_latent.register_forward_hook(get_activation(layer_remark)))                
            else:
                layer_remark = 'avgpool'
                hooker_handles.append(model.avgpool.register_forward_hook(get_activation(layer_remark)))

            model(data)
            [h.remove() for h in hooker_handles]
            feature = activation[layer_remark]

            dim = feature.shape[1]
            activation_log.append(feature.data.cpu().numpy().reshape(curr_batch_size, dim, -1).mean(2))

        count += len(data)

    activation_log = np.concatenate(activation_log, axis=0)

    return np.percentile(activation_log.flatten(), 90)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.stools import compute_GMM_score

param_dict = {
    "cifar-10": {
        "resnet18": 0.01,
        "resnet18_cl1.0": 0.07,
        "resnet50": 0.005,
        "wrn":0.01,
        "vit":0.01,
    },
    "cifar-100": {
        "resnet18": 0.04,
        "resnet18_cl1.0": 0.04,
        "resnet50": 0.005,
        "wrn":0.04,
        "vit":0.04,
    },
    "imagenet":{
        "resnet18": 0.005,
        "resnet50": 0.005,
        "resnet50_cl1.0": 0.0,
        "mobilenet": 0.03,
        "mobilenet_cl1.3": 0.04,
        "vit":0.01,
    }
}


def get_mls_score(inputs, model, forward_func, logits=None, using_ECE=False):
    if logits is None:
        with torch.no_grad():
            feat, logits = forward_func(inputs, model)

    scores = np.max(logits.data.cpu().numpy(), axis=1)
    if using_ECE:
        scores = logits.data.cpu().numpy() / scores.reshape(-1, 1)
    return scores


def get_msp_score(inputs, model, forward_func, logits=None, using_ECE=False):
    if logits is None:
        with torch.no_grad():
            feat, logits = forward_func(inputs, model)
    output_soft = F.softmax(logits, dim=1).detach().cpu().numpy()
    scores = np.max(output_soft, axis=1)
    if using_ECE:
        scores = output_soft
    return scores


def get_energy_score(inputs, model, forward_func, logits=None, using_ECE=False):
    if logits is None:
        with torch.no_grad():
            feat, logits = forward_func(inputs, model)
    
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    if using_ECE:
        scores = torch.log(torch.exp(logits)).data.cpu().numpy() / scores.reshape(-1, 1)
    return scores


def get_odin_score(inputs, model, forward_func, criterion, options):
    name = options['in_dataset']
    if 'cifar-10' in options['in_dataset']:
        if options['in_dataset'] == 'cifar-10-10':
            name = "cifar-10"
        elif options['in_dataset'] == 'cifar-10-100':
            name = "cifar-100"
    elif options['in_dataset'] in ['imagenet', 'tinyimagenet', 'cub', 'waterbird' 'scars', 'aircraft']:
        name = "imagenet"

    # Scaling values according to original ODIN code
    # if name == 'cifar-10':
    #     normalized_scalars = (0.2023, 0.1994, 0.2010)
    # elif name == 'cifar-100':
    #     normalized_scalars = (0.2675, 0.2565, 0.2761)
    # elif name == "imagenet":
    #     normalized_scalars = (0.229, 0.224, 0.225)


    model_name = options['model']
    if options['model'] == 'timm_resnet50_pretrained':
        model_name = 'resnet50'
    elif 'dino_vit' in options['model']:
        model_name = 'vit'
    temper = 1000.0
    # noiseMagnitude1 = param_dict[name][model_name]
    noiseMagnitude1 = options['magnitude']

    inputs.requires_grad = True
    model.zero_grad()

    # with torch.enable_grad():
    feat, outputs = forward_func(inputs, model)
    outputs, _ = criterion(feat, outputs)

    labels = outputs.detach().argmax(axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    _, loss = criterion(feat, outputs, labels=labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.detach(), 0)
    gradient = (gradient.float() - 0.5) * 2

    # Scaling values taken from original ODIN code
    # gradient[:, 0] = (gradient[:, 0]) / normalized_scalars[0]
    # gradient[:, 1] = (gradient[:, 1]) / normalized_scalars[1]
    # gradient[:, 2] = (gradient[:, 2]) / normalized_scalars[2]

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.detach(), gradient, alpha=-noiseMagnitude1)
    _, outputs = forward_func(tempInputs, model)
    outputs = outputs / temper

    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu().numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    scores = np.max(nnOutputs, axis=1)
    return scores


def get_godin_score(inputs, model, forward_func, options):
    name = options['in_dataset']
    if 'cifar-10' in options['in_dataset']:
        if options['in_dataset'] == 'cifar-10-10':
            name = "cifar-10"
        elif options['in_dataset'] == 'cifar-10-100':
            name = "cifar-100"
    elif options['in_dataset'] in ['imagenet', 'tinyimagenet', 'cub', 'waterbird', 'scars', 'aircraft']:
        name = "imagenet"

    # Scaling values according to original ODIN code
    # if name == 'cifar-10':
    #     normalized_scalars = (0.2023, 0.1994, 0.2010)
    # elif name == 'cifar-100':
    #     normalized_scalars = (0.2675, 0.2565, 0.2761)
    # elif name == "imagenet":
    #     normalized_scalars = (0.229, 0.224, 0.225)


    model_name = options['model']
    if options['model'] == 'timm_resnet50_pretrained':
        model_name = 'resnet50'
    noiseMagnitude1 = param_dict[name][model_name]
    # noiseMagnitude1 = options['magnitude']

    inputs.requires_grad = True
    model.zero_grad()

    dist, _ = forward_func(inputs, model)
    max_scores, _ = torch.max(dist, dim = 1)
    max_scores.backward(torch.ones(len(max_scores)).to(inputs.device))

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.detach(), 0)
    gradient = (gradient.float() - 0.5) * 2

    # Scaling values taken from original ODIN code
    # gradient[:, 0] = (gradient[:, 0]) / normalized_scalars[0]
    # gradient[:, 1] = (gradient[:, 1]) / normalized_scalars[1]
    # gradient[:, 2] = (gradient[:, 2]) / normalized_scalars[2]

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.detach(), gradient, alpha=-noiseMagnitude1)
    dist, _ = forward_func(tempInputs, model)
    scores = torch.max(dist, dim=1)[0].data.cpu().numpy()
    return scores


def get_geodin_score(inputs, model, forward_func, options):
    with torch.no_grad():
        feat, logits = forward_func(inputs, model)
        feat_norm = torch.norm(feat, dim=1).unsqueeze(1)

    a = nn.Sigmoid()(nn.Linear(options['num_classes'], 1)(logits))
    b = nn.Softplus()(nn.Linear(options['num_classes'], 1)(logits))
    logits = logits/feat_norm * (feat_norm/a + b/a)

    pred = torch.nn.functional.softmax(logits,dim=1)
    scores = pred.max(1)[0].cpu().numpy()
    return scores


def get_gradnorm_score(inputs, model, options):
    scores = []

    logsoftmax = torch.nn.LogSoftmax(dim=-1)

    model.zero_grad()
    
    B = inputs.size(0)

    for i in range(B):
        data = inputs[i, :].unsqueeze(0)

        with torch.enable_grad():
            _, outputs = model(data)

            targets = torch.ones((1, options['num_classes'])).to(options['device'])
            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
            loss.backward()
            if options['in_dataset'] in ['cub', 'scars', 'aircraft']:
                layer_grad = model.resnet.fc.weight.grad.data.cpu().numpy()
            else:
                try:
                    layer_grad = model.fc.weight.grad.data.cpu().numpy()
                except:
                    layer_grad = model.linear.weight.grad.data.cpu().numpy()

        layer_grad_norm = np.sum(np.abs(layer_grad), axis=1)
        scores.append(layer_grad_norm)

    return np.concatenate(scores, 0)


def get_sem_score(inputs, model, options):
    alpha_list = [-0.0001, 0, 0, 0, 1]
    num_layer = len(options['feature_type_list'])

    for layer_index in range(num_layer):
        score = compute_GMM_score(model, inputs,
                                  options['feature_mean'], options['feature_prec'], options['component_weight_list'], options['transform_matrix_list'],
                                  layer_index, options['feature_type_list'])
        if layer_index == 0:
            score_list = score.view([-1, 1])
        else:
            score_list = torch.cat((score_list, score.view([-1, 1])), 1)
    alpha = torch.cuda.FloatTensor(alpha_list)
    scores = torch.matmul(torch.log(score_list + 1e-45), alpha)
    return scores.data.cpu().numpy()


def get_score(inputs, model, forward_threshold, criterion, options, logits, using_ECE=False):
    cmd_list = options['ood_method'].split('_')
    cmd_len = len(cmd_list)
    cmd = cmd_list[0] if cmd_len == 1 else cmd_list[1]

    if cmd == 'msp':
        scores = get_msp_score(inputs, model, forward_threshold, logits=logits, using_ECE=using_ECE)
    elif cmd == 'mls':
        scores = get_mls_score(inputs, model, forward_threshold, logits=logits, using_ECE=using_ECE)
    elif cmd == 'odin':
        scores = get_odin_score(inputs, model, forward_threshold, criterion, options)
    elif cmd == 'energy':
        scores = get_energy_score(inputs, model, forward_threshold, logits=logits, using_ECE=using_ECE)
    elif cmd == 'godin':
        scores = get_godin_score(inputs, model, forward_threshold, options)    
    elif cmd == 'geodin':
        scores = get_geodin_score(inputs, model, forward_threshold, options)
    elif cmd == 'gradnorm':
        scores = get_gradnorm_score(inputs, model, options)
    elif cmd == 'sem':
        scores = get_sem_score(inputs, model, options)    
    return scores
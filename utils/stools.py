import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy import linalg
from sklearn.covariance import (empirical_covariance, ledoit_wolf,
                                shrunk_covariance)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def tensor2list(x):
    return x.data.cpu().tolist()


def get_torch_feature_stat(feature, only_mean=False):
    feature = feature.view([feature.size(0), feature.size(1), -1])
    feature_mean = torch.mean(feature, dim=-1)
    feature_var = torch.var(feature, dim=-1)
    if feature.size(-2) * feature.size(-1) == 1 or only_mean:
        feature_stat = feature_mean
    else:
        feature_stat = torch.cat((feature_mean, feature_var), 1)
    return feature_stat


def process_feature_type(feature_temp, feature_type):
    if feature_type == 'flat':
        feature_temp = feature_temp.view([feature_temp.size(0), -1])
    elif feature_type == 'stat':
        feature_temp = get_torch_feature_stat(feature_temp)
    elif feature_type == 'mean':
        feature_temp = get_torch_feature_stat(feature_temp, only_mean=True)
    else:
        raise ValueError('Unknown feature type')
    return feature_temp


def reduce_feature_dim(feature_list_full, label_list_full, feature_process):
    if feature_process == 'none':
        transform_matrix = np.eye(feature_list_full.shape[1])
    else:
        feature_process, kept_dim = feature_process.split('_')
        kept_dim = int(kept_dim)
        if feature_process == 'capca':
            lda = InverseLDA(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        elif feature_process == 'pca':
            pca = PCA(n_components=kept_dim)
            pca.fit(feature_list_full)
            transform_matrix = pca.components_.T
        elif feature_process == 'lda':
            lda = LinearDiscriminantAnalysis(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        else:
            raise Exception('Unknown Process Type')
    return transform_matrix


@torch.no_grad()
def get_GMM_stat(model, trainloader, num_clusters_list, feature_type_list, reduce_dim_list, options):
    """ Compute GMM.
    Args:
        model (nn.Module): pretrained model to extract features
        num_clusters_list (list): number of clusters for each layer
        feature_type_list (list): feature type for each layer
        reduce_dim_list (list): dim-reduce method for each layer
    return: feature_mean: list of class mean
            feature_prec: list of precisions
            component_weight_list: list of component
            transform_matrix_list: list of transform_matrix
    """
    feature_mean_list, feature_prec_list = [], []
    component_weight_list, transform_matrix_list = [], []
    num_layer = len(num_clusters_list)
    feature_all = [None for x in range(num_layer)]
    label_list = []

    # collect features
    for batch in tqdm(trainloader, desc='Compute GMM Stats [Collecting]'):
        if len(batch) == 2:
            data, label = batch
        elif len(batch) == 3:
            data, label, idx = batch
        data, label = data.to(options['device']), label.to(options['device'])

        feature_list = model.feature_list(data)
        label_list.extend(tensor2list(label))
        for layer_idx in range(num_layer):
            feature_type = feature_type_list[layer_idx]
            feature_processed = process_feature_type(feature_list[layer_idx], feature_type)
            if isinstance(feature_all[layer_idx], type(None)):
                feature_all[layer_idx] = tensor2list(feature_processed)
            else:
                feature_all[layer_idx].extend(tensor2list(feature_processed))
    label_list = np.array(label_list)

    # reduce feature dim and perform gmm estimation
    for layer_idx in tqdm(range(num_layer), desc='Compute GMM Stats [Estimating]'):
        feature_sub = np.array(feature_all[layer_idx])
        transform_matrix = reduce_feature_dim(feature_sub, label_list, reduce_dim_list[layer_idx])
        feature_sub = np.dot(feature_sub, transform_matrix)
        # GMM estimation
        gm = GaussianMixture(n_components=num_clusters_list[layer_idx], random_state=0, covariance_type='tied').fit(feature_sub)
        feature_mean = gm.means_
        feature_prec = gm.precisions_
        component_weight = gm.weights_

        feature_mean_list.append(torch.Tensor(feature_mean).cuda())
        feature_prec_list.append(torch.Tensor(feature_prec).cuda())
        component_weight_list.append(torch.Tensor(component_weight).cuda())
        transform_matrix_list.append(torch.Tensor(transform_matrix).cuda())

    return feature_mean_list, feature_prec_list, component_weight_list, transform_matrix_list


def compute_GMM_score(model, data, feature_mean, feature_prec, component_weight, transform_matrix, layer_idx, feature_type_list):
    """ Compute GMM.
    Args:
        model (nn.Module): pretrained model to extract features
        data (DataLoader): input one training batch
        feature_mean (list): a list of torch.cuda.Tensor()
        feature_prec (list): a list of torch.cuda.Tensor()
        component_weight (list): a list of torch.cuda.Tensor()
        transform_matrix (list): a list of torch.cuda.Tensor()
        layer_idx (int): index of layer in interest
        feature_type_list (list): a list of strings to indicate feature type
        return_pred (bool): return prediction and confidence, or only conf.
    return:
        pred (torch.cuda.Tensor):
        prob (torch.cuda.Tensor):
    """
    # extract features
    feature_list = model.feature_list(data)
    feature_list = process_feature_type(feature_list[layer_idx], feature_type_list[layer_idx])
    feature_list = torch.mm(feature_list, transform_matrix[layer_idx])

    # compute prob
    for cluster_idx in range(len(feature_mean[layer_idx])):
        zero_f = feature_list - feature_mean[layer_idx][cluster_idx]
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, feature_prec[layer_idx]), zero_f.t()).diag()
        prob_gau = torch.exp(term_gau)
        if cluster_idx == 0:
            prob_matrix = prob_gau.view([-1, 1])
        else:
            prob_matrix = torch.cat((prob_matrix, prob_gau.view(-1, 1)), 1)

    prob = torch.mm(prob_matrix, component_weight[layer_idx].view(-1, 1))
    return prob


def get_Mahalanobis_scores(model, test_loader, num_classes, sample_mean,
                           precision, transform_matrix, layer_index,
                           feature_type_list, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    for batch in tqdm(test_loader,
                      desc=f'{test_loader.dataset.name}_layer{layer_index}'):
        data = batch['data'].cuda()
        data = Variable(data, requires_grad=True)
        noise_gaussian_score = compute_Mahalanobis_score(
            model, data, num_classes, sample_mean, precision, transform_matrix,
            layer_index, feature_type_list, magnitude)
        Mahalanobis.extend(noise_gaussian_score.detach().cpu().numpy())
    return Mahalanobis


def compute_Mahalanobis_score(model,
                              data,
                              num_classes,
                              sample_mean,
                              precision,
                              transform_matrix,
                              layer_index,
                              feature_type_list,
                              magnitude,
                              return_pred=False):
    # extract features
    _, out_features = model(data, return_feature_list=True)
    out_features = process_feature_type(out_features[layer_index],
                                        feature_type_list[layer_index])
    out_features = torch.mm(out_features, transform_matrix[layer_index])

    # compute Mahalanobis score
    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]),
                                   zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)),
                                       1)

    # Input_processing
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)
    pure_gau = -0.5 * torch.mm(
        torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()

    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # here we use the default value of 0.5
    gradient.index_copy_(
        1,
        torch.LongTensor([0]).cuda(),
        gradient.index_select(1,
                              torch.LongTensor([0]).cuda()) / 0.5)
    gradient.index_copy_(
        1,
        torch.LongTensor([1]).cuda(),
        gradient.index_select(1,
                              torch.LongTensor([1]).cuda()) / 0.5)
    gradient.index_copy_(
        1,
        torch.LongTensor([2]).cuda(),
        gradient.index_select(1,
                              torch.LongTensor([2]).cuda()) / 0.5)
    tempInputs = torch.add(
        data.data, gradient,
        alpha=-magnitude)  # updated input data with perturbation

    with torch.no_grad():
        _, noise_out_features = model(Variable(tempInputs),
                                      return_feature_list=True)
        noise_out_features = process_feature_type(
            noise_out_features[layer_index], feature_type_list[layer_index])
        noise_out_features = torch.mm(noise_out_features,
                                      transform_matrix[layer_index])

    noise_gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]),
                                   zero_f.t()).diag()
        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat(
                (noise_gaussian_score, term_gau.view(-1, 1)), 1)

    noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
    if return_pred:
        return sample_pred, noise_gaussian_score
    else:
        return noise_gaussian_score


def ash_b(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x
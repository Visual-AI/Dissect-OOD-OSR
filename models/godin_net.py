import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Derived from https://github.com/guyera/Generalized-ODIN-Implementation/blob/master/code/deconfnet.py
"""
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

class Cosine(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Cosine, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias=False)
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity="relu")

    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight).to(x.device)
        ret = (torch.matmul(x, w.T))
        return ret


class Euclidean(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Euclidean, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        x = x.unsqueeze(2) #(batch, latent, 1)
        h = self.h.weight.T.unsqueeze(0) #(1, latent, num_classes)
        ret = -((x -h).pow(2)).mean(1)
        return ret

        
class Inner(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Inner, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        return self.h(x)


class UpperNet(nn.Module):
    def __init__(self, base_model, in_features, num_classes):
        super(UpperNet, self).__init__()
        self.base_model = base_model
        self.dist = Cosine(in_features, num_classes)
        self.g = nn.Sequential(nn.Linear(in_features, 1), nn.BatchNorm1d(1), nn.Sigmoid())

    def forward(self, x, return_feat=False, domain_label=None):
        if domain_label is None:
            domain_label = 0 * torch.ones(x.shape[0], dtype=torch.long).to(x.device)
        feat, _ = self.base_model(x, True)
        ret = self.dist(feat)
        logits = ret / self.g.to(feat.device)(feat)
        if return_feat:
            return feat, logits
        else:
            return logits

    def forward_threshold(self, x, threshold=1e10, return_feat=False):
        feat, _ = self.base_model(x, True)
        feat = feat.clip(max=threshold)
        ret = self.dist(feat)
        logits = ret / self.g.to(feat.device)(feat)
        if return_feat:
            return ret, logits
        else:
            return logits
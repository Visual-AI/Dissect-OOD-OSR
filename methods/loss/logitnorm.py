import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitNormLoss(nn.Module):
    def __init__(self, t=1.0, **options):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, y, labels):
        logits = y

        if labels is None:
            return logits, 0

        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(logits, norms) / self.t
        
        return F.cross_entropy(logit_norm, labels)
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.loss.LabelSmoothing import smooth_cross_entropy_loss

class OELoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(OELoss, self).__init__()
        self.temp = options['temp']
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, length, labels):
        if not self.label_smoothing:
            loss = F.cross_entropy(x[:length] / self.temp, labels)
        else:
            loss = smooth_cross_entropy_loss(x[:length] / self.temp, labels=labels, smoothing=self.label_smoothing, dim=-1)

        # cross-entropy from softmax distribution to uniform distribution
        loss += 0.5 * -(x[length:].mean(1) - torch.logsumexp(x[length:], dim=1)).mean()
        
        return x, loss

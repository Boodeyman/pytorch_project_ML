# В src/loss/example.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ASoftmaxLoss(nn.Module):
    def __init__(self, in_feats=32, n_classes=2, m=4, s=30.0):
        super().__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.m = m
        self.s = s

        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_feats))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, logits, labels, **batch):
        # A-Softmax implementation
        cosine = F.linear(F.normalize(logits), F.normalize(self.weight))
        phi = cosine - self.m

        one_hot = torch.zeros(cosine.size()).to(cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, labels)
        return {"loss": loss}

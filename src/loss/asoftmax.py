import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ASoftmaxLoss(nn.Module):
    """
    Angular Softmax Loss (A-Softmax) для anti-spoofing
    Основано на STC paper для ASVspoof 2019
    """

    def __init__(self, in_features=128, num_classes=2, m=4, s=30.0):
        """
        Args:
            in_features (int): размер входных features
            num_classes (int): количество классов
            m (int): angular margin multiplier
            s (float): feature scale
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = m
        self.s = s

        # Learnable weights for each class
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model outputs [B, in_features]
            labels (Tensor): ground truth labels [B]
        Returns:
            dict: {"loss": loss_value}
        """
        # Normalize features and weights
        features_norm = F.normalize(logits, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(features_norm, weight_norm)

        # Compute angular margin
        # cos(m*theta) for the correct class
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)

        # Create one-hot encoding
        one_hot = torch.zeros(cosine.size()).to(cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply angular margin only to the correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)

        return {"loss": loss}

#%%
"""
LCVaR uses the original implementation from http://proceedings.mlr.press/v119/xu20b/xu20b.pdf
"""
from typing import List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LHCVaRLoss(nn.Module):
    def __init__(self, alpha: torch.tensor, strat='max'):
        super(LHCVaRLoss, self).__init__()
        self.alpha = nn.Parameter(alpha, requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(10., dtype=torch.float),
                                      requires_grad=True)

        self.prev_loss = {}
        self.strat = strat

    def forward(self, losses, labels):
        device = losses.device

        prev_loss = {}
        inner_term = 0
        label_set = labels.unique()
        mean_class_losses = []
        class_cts = []
        for label in label_set:
            label_p = torch.mean((labels == label).float())
            class_losses = losses[labels == label]
            mean_class_loss = torch.mean(class_losses)
            mean_class_losses.append(mean_class_loss)
            class_cts.append(label_p)

        mean_class_losses = torch.stack(mean_class_losses)
        class_cts = torch.stack(class_cts)

        c_losses = []
        for idx in range(label_set.shape[0]):
            mcl = mean_class_losses[idx].item()
            self.prev_loss[label_set[idx].item()] = mcl
            c_losses.append(mcl)

        if self.strat == 'analytic':
            weight_sum = 0
            best_lambda = 0
            factors = class_cts * self.alpha[label_set]
            for idx, loss in sorted(enumerate(c_losses), key=lambda x: -x[1]):
                if weight_sum + factors[idx].item() >= 1:
                    best_lambda = loss
                else:
                    weight_sum += factors[idx].item()
            self.threshold.data = torch.tensor(best_lambda,
                                               dtype=torch.float,
                                               device=self.threshold.device)
        else:
            max_mean_class_loss = torch.max(mean_class_losses).item()
            if self.threshold.item() > max_mean_class_loss:
                self.threshold.data = torch.tensor(
                    max_mean_class_loss,
                    dtype=torch.float,
                    device=self.threshold.device)
        inner_term = class_cts * F.relu(mean_class_losses -
                                        self.threshold) / self.alpha[label_set]
        return torch.sum(inner_term) + self.threshold * losses.shape[0]


class LCVaRLoss(LHCVaRLoss):
    def __init__(self,
                 classes: int,
                 alpha: float,
                 strat='max'):
        alphas = torch.zeros(classes).fill_(alpha)
        super(LCVaRLoss, self).__init__(alphas, strat)

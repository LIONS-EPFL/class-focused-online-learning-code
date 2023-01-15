"""
Idea:
Stay close to exponentially decaying average of previous predictors.
Do so more and more (so that a tail of past predictors have actually been build).
Then all future training will be stabilized.
"""
import math
import torch
from torch.nn import functional as F
import attr


class TemporalEnsembling(torch.nn.Module):
    """
    Remember that if you restart a model from a previous checkpoint it is extreme important that the training data remains exactly the same (since the indices would otherwise be scrampled).
    """
    def __init__(self,
        momentum=0.9,
        weight=30.0,
        rampup_min_epoch=0,
        rampup_max_epoch=50,
        num_classes=10,
        num_samples=40_000
    ):
        super().__init__()

        self.momentum = momentum
        self.weight = weight
        self.rampup_min_epoch = rampup_min_epoch
        self.rampup_max_epoch = rampup_max_epoch
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.p = torch.nn.Parameter(torch.zeros((num_samples, num_classes)), requires_grad=False)

    def get_weight(self, epoch):
        if epoch < self.rampup_min_epoch:
            return 0.0
        if epoch > self.rampup_max_epoch:
            return 1.0
        else:
            phase = 1.0 - (epoch - self.rampup_min_epoch) / (self.rampup_max_epoch - self.rampup_min_epoch)
            return math.exp(-5.0 * phase * phase)

    def update(self, indices, logits, epoch):
        """
        TODO: Does not work correctly when sampling with replacement (if same index is drawn twice in the same batch)!
        """
        if epoch >= self.rampup_min_epoch:
            preds = F.softmax(logits, dim=-1)
            self.p[indices] = (self.momentum * self.p[indices] + (1-self.momentum) * preds).detach()

    def forward(self, epoch, indices, logits, reduction='mean'):
        preds = F.softmax(logits, dim=-1)
        reg_term = self.get_weight(epoch) * ((self.p[indices] - preds) ** 2).sum(-1)
        
        if reduction == 'mean':
            return reg_term.mean()
        elif reduction == 'none':
            return reg_term
        else:
            raise ValueError("Only mean and none supported")
